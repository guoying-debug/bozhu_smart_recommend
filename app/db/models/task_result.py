"""
AnalysisTask —— Celery 任务的业务结果持久化表
"""
from sqlalchemy import Column, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.db.base import Base


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    task_id = Column(String(64), primary_key=True)
    status = Column(String(20), nullable=False, default="PENDING")  # PENDING/SUCCESS/FAILURE
    input_title = Column(Text, nullable=False)
    input_category = Column(String(100))
    result_json = Column(JSON)
    error = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    finished_at = Column(DateTime)

    def __repr__(self):
        return f"<AnalysisTask(id={self.task_id!r}, status={self.status!r})>"
