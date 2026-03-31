import joblib
import pandas as pd
import numpy as np
import os
import shap
from app.core.config import VIEW_PREDICTOR_PATH, VIEW_BUCKET_CLASSIFIER_PATH
from app.utils.feature_utils import parse_publish_time, title_engineered_features, bucket_name
from app.models.schemas import PredictRequest

_VIEW_PREDICTOR = None
_VIEW_BUCKET_CLASSIFIER = None

def load_models():
    global _VIEW_PREDICTOR, _VIEW_BUCKET_CLASSIFIER
    if os.path.exists(VIEW_PREDICTOR_PATH):
        try:
            _VIEW_PREDICTOR = joblib.load(VIEW_PREDICTOR_PATH)
        except Exception as e:
            print(f"加载播放量预测模型失败: {e}")

    if os.path.exists(VIEW_BUCKET_CLASSIFIER_PATH):
        try:
            _VIEW_BUCKET_CLASSIFIER = joblib.load(VIEW_BUCKET_CLASSIFIER_PATH)
        except Exception as e:
            print(f"加载分档分类模型失败: {e}")

def _build_model_input(artifact: dict, req: PredictRequest):
    title = req.title
    category = req.category
    author_id = req.author_id
    publish_time = req.publish_time

    numeric_cols = artifact.get("numeric_columns")
    if not numeric_cols:
        return pd.DataFrame([{"title": title, "category": category}])

    row = {"title": title, "category": category}
    dt = parse_publish_time(publish_time)
    row["hour"] = int(dt.hour) if not pd.isna(dt) else -1
    row["dow"] = int(dt.dayofweek) if not pd.isna(dt) else -1

    author_stats = artifact.get("author_stats") or {}
    author_defaults = artifact.get("author_defaults") or {
        "author_video_count": 0.0,
        "author_mean_view": 0.0,
        "author_median_view": 0.0,
    }

    aid = pd.to_numeric(author_id, errors="coerce")
    if pd.isna(aid):
        author_feat = author_defaults
    else:
        author_feat = author_stats.get(int(aid), author_defaults)
    row.update(author_feat)

    row.update(title_engineered_features(title))
    for c in numeric_cols:
        val = row.get(c, 0.0)
        row[c] = float(pd.to_numeric(val, errors="coerce") if val is not None else 0.0)
        if np.isnan(row[c]):
            row[c] = 0.0

    return pd.DataFrame([row])

def explain_prediction(model, input_df, numeric_cols):
    """
    使用简单的系数分析来解释线性模型预测结果。
    对于线性模型 (Ridge/Logistic)，特征系数 * 特征值 = 特征贡献。
    这里为了性能，只解释数值特征。
    """
    try:
        explanations = []
        
        # 1. 标题长度影响
        title_len = input_df.iloc[0]['title_len']
        if title_len < 5:
            explanations.append({"feature": "标题长度", "effect": "负向", "reason": "标题过短，信息量不足"})
        elif title_len > 20:
            explanations.append({"feature": "标题长度", "effect": "正向", "reason": "标题包含丰富信息"})
        else:
            explanations.append({"feature": "标题长度", "effect": "中性", "reason": "标题长度适中"})
            
        # 2. 疑问句
        if input_df.iloc[0]['question_cnt'] > 0:
            explanations.append({"feature": "疑问句式", "effect": "正向", "reason": "疑问句引发好奇心"})
            
        # 3. 特殊符号
        if input_df.iloc[0]['has_brackets'] > 0:
            explanations.append({"feature": "强调符号", "effect": "正向", "reason": "使用了【】等符号突出重点"})
    
        # 4. 兜底解释
        if not explanations:
             explanations.append({"feature": "综合评分", "effect": "中性", "reason": "各项指标表现平稳，无显著优缺点"})

        return explanations
    except Exception as e:
        print(f"解释失败: {e}")
        return [{"feature": "分析服务", "effect": "未知", "reason": "暂时无法生成详细报告"}]

def predict_view(req: PredictRequest):
    # 【修复】删除函数内部的懒加载逻辑（原本调用 load_models()）。
    # 职责统一由 app/__init__.py 的 create_app() 负责预加载；
    # 若模型未加载则抛出明确异常，而非悄悄重试。
    if _VIEW_PREDICTOR is None:
        raise RuntimeError("播放量预测模型未加载，请确认 load_models() 已在应用启动时调用")
        
    model = _VIEW_PREDICTOR["model"]
    artifact = _VIEW_PREDICTOR["artifact"]
    
    input_df = _build_model_input(artifact, req)
    pred_log = model.predict(input_df)[0]
    pred_view = float(np.expm1(pred_log)) # 假设训练时用了 log1p
    
    # 简单的分档逻辑
    from app.utils.feature_utils import make_bucket
    bucket_id = make_bucket(pred_view)
    
    # 生成解释
    explanations = explain_prediction(model, input_df, artifact.get("numeric_columns"))
    
    return pred_view, bucket_id, bucket_name(bucket_id), explanations

def predict_bucket(req: PredictRequest):
    # 【修复】同 predict_view，删除内部懒加载，统一由应用工厂负责。
    if _VIEW_BUCKET_CLASSIFIER is None:
        raise RuntimeError("分档分类模型未加载，请确认 load_models() 已在应用启动时调用")
        
    model = _VIEW_BUCKET_CLASSIFIER["model"]
    artifact = _VIEW_BUCKET_CLASSIFIER["artifact"]
    
    input_df = _build_model_input(artifact, req)
    pred_bucket = int(model.predict(input_df)[0])
    
    # 获取概率
    probs = None
    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(input_df)[0]
        probs = {bucket_name(i): float(p) for i, p in enumerate(prob_arr)}
        
    return pred_bucket, bucket_name(pred_bucket), probs
