from app.skills.base import BaseSkill
from app.core.predictor import predict_view
from app.models.schemas import PredictRequest


class PlayPredictSkill(BaseSkill):
    name = "播放量预测工具"
    description = "当用户需要预测某个标题的B站播放量时调用。输入视频标题"

    def run(self, title: str) -> str:
        try:
            pred_view, _, bucket_name, explanations = predict_view(PredictRequest(title=title, category="未知"))
            features = "\n".join(f"- {e['feature']}: {e['effect']} ({e['reason']})" for e in explanations)
            return f"预估播放量：{int(pred_view):,} 次\n所属档位：{bucket_name}\n\n特征分析：\n{features}"
        except Exception as e:
            return f"预测失败: {e}"
