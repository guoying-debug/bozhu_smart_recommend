"""
Celery 应用实例
"""
from celery import Celery
from app.core.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND


def make_celery(app=None):
    """创建 Celery 实例，可选择绑定 Flask app context"""
    celery = Celery(
        "bilibili",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=[
            "app.tasks.title_tasks",
            "app.tasks.data_tasks",
        ],
    )
    celery.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="Asia/Shanghai",
        enable_utc=True,
        task_track_started=True,
        # Beat 调度表（阶段四步骤11）
        beat_schedule={
            "daily-clustering": {
                "task": "app.tasks.data_tasks.run_clustering_task",
                "schedule": {"hour": 2, "minute": 0},  # 每天凌晨2点
            }
        },
    )

    if app is not None:
        # 让 task 内部能通过 app_context 访问 Flask/SQLAlchemy
        class ContextTask(celery.Task):
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)

        celery.Task = ContextTask

    return celery


# 模块级单例（worker / beat 进程直接 import 使用）
celery = make_celery()
