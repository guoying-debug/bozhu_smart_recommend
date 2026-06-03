from flask import Blueprint, jsonify, request, render_template
from app.core.analysis import get_cluster_summary, get_topics_list, get_feature_importance
from app.core.predictor import predict_view, predict_bucket
from app.core.recommender import get_similar_titles
from app.core.llm import analyze_title_with_llm
from app.models.schemas import PredictRequest, PredictViewResponse, PredictBucketResponse, ErrorResponse, TaskSubmitResponse, TaskStatusResponse
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
    """标题建议（异步版）— 提交任务，返回 task_id"""
    data = request.json or {}
    title = data.get("title", "")
    category = data.get("category", "未分类")

    if not title:
        return jsonify({"error": "Title is required"}), 400

    from app.tasks.title_tasks import analyze_title_task
    from app.db.session import get_session
    from app.db.models.task_result import AnalysisTask

    # 提交 Celery 任务
    job = analyze_title_task.delay(title, category)

    # 预先在 MySQL 创建 PENDING 记录，方便历史查询
    with get_session() as session:
        session.add(AnalysisTask(
            task_id=job.id,
            status="PENDING",
            input_title=title,
            input_category=category,
        ))

    return jsonify(TaskSubmitResponse(task_id=job.id).model_dump()), 202


@api_bp.route('/task/<task_id>', methods=['GET'])
def api_task_status(task_id):
    """轮询任务状态和结果"""
    from app.celery_app import celery
    from app.db.session import get_session
    from app.db.models.task_result import AnalysisTask

    # 优先读 MySQL 业务结果
    with get_session() as session:
        record = session.query(AnalysisTask).filter_by(task_id=task_id).first()

    if record and record.status in ("SUCCESS", "FAILURE"):
        return jsonify(TaskStatusResponse(
            task_id=task_id,
            status=record.status,
            result=record.result_json,
            error=record.error,
        ).model_dump())

    # 回退到 Celery backend 查状态
    job = celery.AsyncResult(task_id)
    status = job.state  # PENDING / STARTED / SUCCESS / FAILURE
    result = job.result if job.state == "SUCCESS" else None
    error = str(job.result) if job.state == "FAILURE" else None

    return jsonify(TaskStatusResponse(
        task_id=task_id,
        status=status,
        result=result,
        error=error,
    ).model_dump())

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
