"""
models 子包 —— 导入所有 model 使 Alembic autogenerate 能发现它们
"""
from app.db.models.video import Video
from app.db.models.task_result import AnalysisTask

__all__ = ["Video", "AnalysisTask"]
