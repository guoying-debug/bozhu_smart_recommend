from flask import Blueprint, jsonify, request, render_template
from app.core.analysis import get_cluster_summary, get_topics_list, get_feature_importance
from app.core.predictor import predict_view, predict_bucket
from app.core.recommender import get_similar_titles
from app.core.llm import analyze_title_with_llm
from app.models.schemas import PredictRequest, PredictViewResponse, PredictBucketResponse, ErrorResponse
from pydantic import ValidationError
import plotly.graph_objects as go
import plotly.express as px
from app.core.data_loader import load_data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/topics', methods=['GET'])
def get_topics():
    """获取话题簇摘要"""
    topics = get_topics_list()
    return jsonify(topics)

@api_bp.route('/analysis_summary', methods=['GET'])
def get_analysis_summary():
    """获取聚类分析摘要"""
    summary = get_cluster_summary()
    return jsonify(summary)

@api_bp.route('/predict_view', methods=['POST'])
def api_predict_view():
    """预测播放量"""
    try:
        data = request.json
        req = PredictRequest(**data)
        
        pred_view, bucket_id, bucket_name, explanations = predict_view(req)
        
        return jsonify({
            "predicted_view": pred_view,
            "predicted_bucket": bucket_name,
            "bucket_id": bucket_id,
            "explanations": explanations
        })
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/predict_bucket', methods=['POST'])
def api_predict_bucket():
    """预测分档"""
    try:
        data = request.json
        req = PredictRequest(**data)
        
        bucket_id, bucket_name, probs = predict_bucket(req)
        
        return jsonify({
            "predicted_bucket": bucket_name,
            "bucket_id": bucket_id,
            "probabilities": probs
        })
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/title_advice', methods=['POST'])
def api_title_advice():
    """标题建议 (AI 增强版：结合预测模型与 LLM 智能体)"""
    try:
        data = request.json
        title = data.get("title", "")
        category = data.get("category", "未分类")
        
        if not title:
            return jsonify({"error": "Title is required"}), 400
            
        # 1. 获取相似爆款 (RAG - 检索增强)
        similar_titles = get_similar_titles(title)
        
        # 2. 调用预测模型获取定量特征分析
        req = PredictRequest(title=title, category=category)
        pred_view, _, _, feature_explanations = predict_view(req)
        
        # 3. 调用 LLM 智能体进行综合诊断 (Agentic Workflow)
        feature_importance = get_feature_importance()
        llm_result = analyze_title_with_llm(
            title=title,
            category=category,
            predicted_view=pred_view,
            feature_explanations=feature_explanations,
            similar_titles=similar_titles,
            feature_importance=feature_importance
        )
        
        # 4. 组装结果
        # 如果 LLM 调用失败或未配置 Key，回退到规则引擎建议
        advice_list = llm_result.get("suggestions", [])
        if not advice_list:
            # 规则引擎兜底
            if len(title) < 10:
                advice_list.append("标题偏短，建议增加描述性词汇或悬念（如：'竟然...'，'这几点...'）")
            if not any(x in title for x in ['【', '】', '！', '？']):
                advice_list.append("建议使用【】突出重点，或使用？！增强语气")
        
        diagnosis = llm_result.get("diagnosis", "智能体正在休息，仅提供基础建议。")

        return jsonify({
            "similar_titles": similar_titles,
            "advice_list": advice_list,
            "diagnosis": diagnosis, # 新增诊断字段
            "summary": f"为您找到 {len(similar_titles)} 个相似热门标题，建议结合参考。"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/visualize/kmeans', methods=['GET'])
def visualize_kmeans():
    """动态生成 K-Means 聚类 t-SNE 可视化"""
    try:
        df = load_data()
        if df is None or 'tsne_x' not in df.columns:
            return jsonify({"error": "数据未加载或缺少聚类坐标"}), 500

        # 使用正确的列名
        cluster_col = 'bert_kmeans_cluster' if 'bert_kmeans_cluster' in df.columns else 'Cluster_Label'

        fig = px.scatter(
            df, x='tsne_x', y='tsne_y', color=cluster_col,
            hover_data=['title', 'view_count'],
            title='B站热门话题聚类分布 (t-SNE)',
            labels={cluster_col: '话题簇', 'tsne_x': 't-SNE 维度1', 'tsne_y': 't-SNE 维度2'}
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/visualize/feature_importance', methods=['GET'])
def visualize_feature_importance():
    """动态生成特征重要性图"""
    try:
        importance = get_feature_importance()
        if not importance:
            return jsonify({"error": "特征重要性数据未加载"}), 500

        df_imp = pd.DataFrame(importance).head(15)
        fig = px.bar(
            df_imp, x='score', y='feature', orientation='h',
            title='播放量预测模型 - 特征重要性 Top 15',
            labels={'score': '重要性分数', 'feature': '特征名称'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
