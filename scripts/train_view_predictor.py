import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import logging
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 添加项目根目录到 sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.core.config import get_database_url, VIEW_PREDICTOR_PATH, VIEW_BUCKET_CLASSIFIER_PATH, MODELS_TRAINED_DIR
from app.utils.feature_utils import parse_publish_time, title_engineered_features, make_bucket
from app.utils.text_utils import jieba_tokenize

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _build_author_stats(df: pd.DataFrame):
    if "author_id" not in df.columns:
        return {}, {"author_mean_view": 0.0, "author_median_view": 0.0, "author_video_count": 0.0}

    tmp = df[["author_id", "view_count"]].copy()
    tmp["author_id"] = pd.to_numeric(tmp["author_id"], errors="coerce")
    tmp = tmp.dropna(subset=["author_id"])
    if tmp.empty:
        return {}, {"author_mean_view": 0.0, "author_median_view": 0.0, "author_video_count": 0.0}

    stats = tmp.groupby("author_id")["view_count"].agg(["mean", "median", "count"]).reset_index()
    stats.columns = ["author_id", "author_mean_view", "author_median_view", "author_video_count"]
    
    defaults = {
        "author_mean_view": float(stats["author_mean_view"].mean()),
        "author_median_view": float(stats["author_median_view"].median()),
        "author_video_count": float(stats["author_video_count"].mean()),
    }
    
    author_stats_map = stats.set_index("author_id").to_dict(orient="index")
    return author_stats_map, defaults

def train_view_predictor():
    logger.info("开始训练播放量预测模型...")
    database_url = get_database_url()
    if not database_url:
        logger.error("未设置数据库连接，跳过训练。")
        return

    engine = create_engine(database_url)
    try:
        df = pd.read_sql("SELECT * FROM videos", engine)
    except Exception as e:
        logger.error(f"读取数据库失败: {e}")
        return

    if df.empty:
        logger.warning("数据库无数据。")
        return

    logger.info(f"加载数据 {len(df)} 条。")

    # 1. 特征工程
    # 时间特征
    df['dt'] = df['publish_time'].apply(parse_publish_time)
    df['hour'] = df['dt'].dt.hour.fillna(-1).astype(int)
    df['dow'] = df['dt'].dt.dayofweek.fillna(-1).astype(int)

    # 标题特征
    title_features = df['title'].apply(title_engineered_features).apply(pd.Series)
    df = pd.concat([df, title_features], axis=1)

    # 作者特征
    author_stats_map, author_defaults = _build_author_stats(df)
    
    def get_author_feat(aid):
        try:
            aid = int(aid)
        except:
            return author_defaults
        return author_stats_map.get(aid, author_defaults)

    author_feat_df = df['author_id'].apply(get_author_feat).apply(pd.Series)
    df = pd.concat([df, author_feat_df], axis=1)

    # 2. 准备数据集
    numeric_features = [
        "hour", "dow", "title_len", "digit_cnt", "question_cnt", "exclam_cnt",
        "has_brackets", "has_percent", "has_colon", "has_tutorial", "has_review",
        "has_list", "has_hot", "emotion_cnt", "emotion_hit_cnt",
        "author_video_count", "author_mean_view", "author_median_view"
    ]
    
    # 填充数值缺失值
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    target_col = "view_count"
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    
    # 目标变换 log1p
    y = np.log1p(df[target_col])
    X = df[["title", "category"] + numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 构建 Pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    text_transformer = TfidfVectorizer(tokenizer=jieba_tokenize, max_features=1000)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, ['category']),
            ('txt', text_transformer, 'title')
        ])

    regressor = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', Ridge(alpha=1.0))])

    # 4. 训练回归模型
    logger.info("正在训练回归模型...")
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    # 评估指标
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae_log = mean_absolute_error(y_test, y_pred)
    mae_real = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    
    metrics = {
        "R2": r2,
        "RMSE_log": rmse,
        "MAE_log": mae_log,
        "MAE_real": mae_real
    }
    logger.info(f"回归模型评估: {metrics}")
    
    # 保存评估结果到 JSON
    with open(os.path.join(MODELS_TRAINED_DIR, 'view_predictor_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 保存回归模型
    artifact = {
        "numeric_columns": numeric_features,
        "author_stats": author_stats_map,
        "author_defaults": author_defaults
    }
    joblib.dump({"model": regressor, "artifact": artifact}, VIEW_PREDICTOR_PATH)
    logger.info(f"回归模型已保存至 {VIEW_PREDICTOR_PATH}")

    # 5. 训练分档分类模型
    logger.info("正在训练分档分类模型...")
    # 构造分档标签
    df['bucket'] = df[target_col].apply(make_bucket)
    y_cls = df['bucket']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    
    classifier = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(max_iter=1000))])
                                 
    classifier.fit(X_train_c, y_train_c)
    
    y_pred_c = classifier.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    f1 = f1_score(y_test_c, y_pred_c, average='weighted')
    
    cls_metrics = {
        "Accuracy": acc,
        "F1_Weighted": f1
    }
    logger.info(f"分类模型评估: {cls_metrics}")
    
    with open(os.path.join(MODELS_TRAINED_DIR, 'view_bucket_classifier_metrics.json'), 'w') as f:
        json.dump(cls_metrics, f, indent=4)
    
    joblib.dump({"model": classifier, "artifact": artifact}, VIEW_BUCKET_CLASSIFIER_PATH)
    logger.info(f"分类模型已保存至 {VIEW_BUCKET_CLASSIFIER_PATH}")

if __name__ == "__main__":
    train_view_predictor()
