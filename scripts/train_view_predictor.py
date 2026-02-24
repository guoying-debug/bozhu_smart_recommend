import os
import re
import time

import jieba
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine


EMOTION_WORDS = [
    "震惊",
    "惊呆",
    "惊了",
    "离谱",
    "离大谱",
    "绝了",
    "太强",
    "逆天",
    "封神",
    "上头",
    "笑死",
    "爆笑",
    "泪目",
    "爆哭",
    "崩溃",
    "翻车",
    "血亏",
    "后悔",
    "破防",
    "治愈",
    "感动",
]

def _get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _jieba_tokenize(text: str):
    return jieba.lcut(text)

def _parse_publish_time(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(int(value), unit="s", errors="coerce")
    return pd.to_datetime(value, errors="coerce")

def _make_bucket(view_count: float):
    if view_count < 100_000:
        return 0
    if view_count < 500_000:
        return 1
    if view_count < 2_000_000:
        return 2
    return 3

def _title_engineered_features(title: str):
    title = "" if pd.isna(title) else str(title)
    emotion_cnt = 0
    emotion_hit_cnt = 0
    for w in EMOTION_WORDS:
        c = title.count(w)
        if c > 0:
            emotion_hit_cnt += 1
            emotion_cnt += c
    return {
        "title_len": len(title),
        "digit_cnt": len(re.findall(r"\d", title)),
        "question_cnt": title.count("?") + title.count("？"),
        "exclam_cnt": title.count("!") + title.count("！"),
        "has_brackets": 1 if any(ch in title for ch in ["【", "】", "[", "]", "（", "）", "(", ")", "《", "》"]) else 0,
        "has_percent": 1 if "%" in title else 0,
        "has_colon": 1 if any(ch in title for ch in [":", "："]) else 0,
        "has_tutorial": 1 if any(k in title for k in ["教程", "教学", "入门", "新手", "一图流", "指南"]) else 0,
        "has_review": 1 if any(k in title for k in ["测评", "评测", "对比", "开箱"]) else 0,
        "has_list": 1 if any(k in title for k in ["盘点", "合集", "汇总", "TOP", "Top", "top"]) else 0,
        "has_hot": 1 if any(k in title for k in ["爆", "最强", "必看", "热门", "全网", "火爆"]) else 0,
        "emotion_cnt": emotion_cnt,
        "emotion_hit_cnt": emotion_hit_cnt,
    }

def _build_author_stats(df: pd.DataFrame):
    if "author_id" not in df.columns:
        return {}, {"author_mean_view": 0.0, "author_median_view": 0.0, "author_video_count": 0.0}

    tmp = df[["author_id", "view_count"]].copy()
    tmp["author_id"] = pd.to_numeric(tmp["author_id"], errors="coerce")
    tmp = tmp.dropna(subset=["author_id"])
    if tmp.empty:
        return {}, {"author_mean_view": 0.0, "author_median_view": 0.0, "author_video_count": 0.0}

    author_group = tmp.groupby("author_id")["view_count"].agg(["count", "mean", "median"]).reset_index()
    stats = {}
    for _, r in author_group.iterrows():
        aid = int(r["author_id"])
        stats[aid] = {
            "author_video_count": float(r["count"]),
            "author_mean_view": float(r["mean"]),
            "author_median_view": float(r["median"]),
        }

    defaults = {
        "author_video_count": float(author_group["count"].mean()),
        "author_mean_view": float(tmp["view_count"].mean()),
        "author_median_view": float(tmp["view_count"].median()),
    }
    return stats, defaults

def _build_feature_frame(df: pd.DataFrame, author_stats: dict, author_defaults: dict):
    base = df.copy()
    base["title"] = base["title"].fillna("").astype(str)
    base["category"] = base.get("category", "未知").fillna("未知").astype(str)

    publish_time = base.get("publish_time")
    if publish_time is None:
        base["hour"] = -1
        base["dow"] = -1
    else:
        dt = publish_time.apply(_parse_publish_time)
        base["hour"] = dt.dt.hour.fillna(-1).astype(int)
        base["dow"] = dt.dt.dayofweek.fillna(-1).astype(int)

    if "author_id" in base.columns:
        base["author_id"] = pd.to_numeric(base["author_id"], errors="coerce")
    else:
        base["author_id"] = np.nan

    def _lookup_author(aid):
        if pd.isna(aid):
            return author_defaults
        aid_int = int(aid)
        return author_stats.get(aid_int, author_defaults)

    author_feat = base["author_id"].apply(_lookup_author).apply(pd.Series)
    base = pd.concat([base, author_feat], axis=1)

    title_feat = base["title"].apply(_title_engineered_features).apply(pd.Series)
    base = pd.concat([base, title_feat], axis=1)

    numeric_cols = [
        "hour",
        "dow",
        "title_len",
        "digit_cnt",
        "question_cnt",
        "exclam_cnt",
        "has_brackets",
        "has_percent",
        "has_colon",
        "has_tutorial",
        "has_review",
        "has_list",
        "has_hot",
        "emotion_cnt",
        "emotion_hit_cnt",
        "author_video_count",
        "author_mean_view",
        "author_median_view",
    ]
    for c in numeric_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

    X = base[["title", "category"] + numeric_cols].copy()
    return X, numeric_cols


def _get_database_url():
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "bilibili_data")
    if not db_password:
        return None
    return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def load_training_data():
    database_url = _get_database_url()
    if database_url:
        engine = create_engine(database_url)
        df = pd.read_sql(
            "SELECT video_id, title, category, author_id, publish_time, view_count FROM videos",
            engine,
        )
        return df

    base_dir = _get_base_dir()
    json_path = os.path.join(base_dir, "src", "bilibili_scraper", "output.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError("未找到数据库连接，也未找到 src/bilibili_scraper/output.json")
    return pd.read_json(json_path, lines=False, encoding="utf-8")


def train_and_save(model_path: str):
    df = load_training_data()
    columns = [c for c in ["title", "category", "author_id", "publish_time", "view_count"] if c in df.columns]
    df = df[columns].copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["category"] = df["category"].fillna("未知").astype(str)
    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce")
    df = df.dropna(subset=["view_count"])
    df = df[df["view_count"] >= 0]

    if len(df) < 50:
        raise ValueError(f"训练数据过少：{len(df)} 条。至少需要 50 条以上。")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    author_stats, author_defaults = _build_author_stats(df_train)
    X_train, numeric_cols = _build_feature_frame(df_train, author_stats, author_defaults)
    X_test, _ = _build_feature_frame(df_test, author_stats, author_defaults)
    y_train = np.log1p(df_train["view_count"].to_numpy())
    y_test = np.log1p(df_test["view_count"].to_numpy())

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "title",
                TfidfVectorizer(
                    tokenizer=_jieba_tokenize,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60000,
                ),
                "title",
            ),
            ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    model = Ridge(alpha=2.0)
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    view_true = np.expm1(y_test)
    view_pred = np.maximum(0.0, np.expm1(y_pred))

    metrics = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae_view": float(mean_absolute_error(view_true, view_pred)),
        "mae_log1p": float(mean_absolute_error(y_test, y_pred)),
        "r2_log1p": float(r2_score(y_test, y_pred)),
    }

    artifact = {
        "pipeline": pipeline,
        "metrics": metrics,
        "trained_at": int(time.time()),
        "feature_columns": ["title", "category"] + numeric_cols,
        "numeric_columns": numeric_cols,
        "author_stats": author_stats,
        "author_defaults": author_defaults,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifact, model_path)
    return metrics


def train_bucket_classifier_and_save(model_path: str):
    df = load_training_data()
    columns = [c for c in ["title", "category", "author_id", "publish_time", "view_count"] if c in df.columns]
    df = df[columns].copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["category"] = df["category"].fillna("未知").astype(str)
    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce")
    df = df.dropna(subset=["view_count"])
    df = df[df["view_count"] >= 0]

    if len(df) < 100:
        raise ValueError(f"训练数据过少：{len(df)} 条。至少需要 100 条以上。")

    y_all = df["view_count"].apply(_make_bucket).astype(int)
    df_train, df_test, y_train, y_test = train_test_split(
        df, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    author_stats, author_defaults = _build_author_stats(df_train)
    X_train, numeric_cols = _build_feature_frame(df_train, author_stats, author_defaults)
    X_test, _ = _build_feature_frame(df_test, author_stats, author_defaults)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "title",
                TfidfVectorizer(
                    tokenizer=_jieba_tokenize,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60000,
                ),
                "title",
            ),
            ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
    )
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "labels": [0, 1, 2, 3],
        "label_names": ["0-10万", "10-50万", "50-200万", "200万+"],
    }

    artifact = {
        "pipeline": pipeline,
        "metrics": metrics,
        "trained_at": int(time.time()),
        "feature_columns": ["title", "category"] + numeric_cols,
        "numeric_columns": numeric_cols,
        "author_stats": author_stats,
        "author_defaults": author_defaults,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifact, model_path)
    return metrics


if __name__ == "__main__":
    base_dir = _get_base_dir()
    reg_path = os.path.join(base_dir, "models", "view_predictor.joblib")
    cls_path = os.path.join(base_dir, "models", "view_bucket_classifier.joblib")

    reg_metrics = train_and_save(reg_path)
    print("回归训练完成，模型已保存：", reg_path)
    print("回归评估指标：", reg_metrics)

    cls_metrics = train_bucket_classifier_and_save(cls_path)
    print("分档训练完成，模型已保存：", cls_path)
    print("分档评估指标：", cls_metrics)
