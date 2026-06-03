"""
标题分析 Celery Task
"""
import logging
from datetime import datetime
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, soft_time_limit=60, name="app.tasks.title_tasks.analyze_title_task")
def analyze_title_task(self, title: str, category: str = "未分类"):
    """
    异步执行标题分析：RAG召回 → 预测模型 → LLM诊断，结果落库。
    """
    from app.db.session import get_session
    from app.db.models.task_result import AnalysisTask

    task_id = self.request.id

    def _update_status(status, result_json=None, error=None):
        with get_session() as session:
            record = session.query(AnalysisTask).filter_by(task_id=task_id).first()
            if record:
                record.status = status
                record.result_json = result_json
                record.error = error
                record.finished_at = datetime.utcnow() if status in ("SUCCESS", "FAILURE") else None

    try:
        self.update_state(state="STARTED")
        _update_status("STARTED")

        # 1. RAG 召回
        try:
            from app.core.recommender import get_similar_titles
            similar_titles = get_similar_titles(title)
        except Exception as e:
            logger.warning(f"RAG召回失败，使用空列表: {e}")
            similar_titles = []

        # 2. 预测模型
        try:
            from app.core.predictor import predict_view
            from app.models.schemas import PredictRequest
            req = PredictRequest(title=title, category=category)
            pred_view, _, _, feature_explanations = predict_view(req)
        except Exception as e:
            logger.warning(f"预测失败，使用默认值: {e}")
            pred_view, feature_explanations = 0, []

        # 3. LLM 诊断
        try:
            from app.core.analysis import get_feature_importance
            from app.core.llm import analyze_title_with_llm
            llm_result = analyze_title_with_llm(
                title=title,
                category=category,
                predicted_view=pred_view,
                feature_explanations=feature_explanations,
                similar_titles=similar_titles,
                feature_importance=get_feature_importance(),
            )
        except Exception as e:
            logger.warning(f"LLM分析失败，使用规则引擎兜底: {e}")
            llm_result = {}

        # 4. 规则引擎兜底
        advice_list = llm_result.get("suggestions", [])
        if not advice_list:
            if len(title) < 10:
                advice_list.append("标题偏短，建议增加描述性词汇或悬念（如：'竟然...'，'这几点...'）")
            if not any(x in title for x in ["【", "】", "！", "？"]):
                advice_list.append("建议使用【】突出重点，或使用？！增强语气")

        result = {
            "similar_titles": similar_titles,
            "advice_list": advice_list,
            "diagnosis": llm_result.get("diagnosis", "智能体正在休息，仅提供基础建议。"),
            "summary": f"为您找到 {len(similar_titles)} 个相似热门标题，建议结合参考。",
        }

        _update_status("SUCCESS", result_json=result)
        return result

    except Exception as exc:
        logger.error(f"analyze_title_task 失败: {exc}")
        _update_status("FAILURE", error=str(exc))
        raise self.retry(exc=exc, countdown=5)
