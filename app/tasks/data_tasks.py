"""
数据相关 Celery Task（聚类、爬取、训练等）
"""
import logging
import subprocess
import os

from celery import shared_task

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_script(script_rel: str, **kwargs) -> dict:
    """辅助：运行 scripts/ 下的脚本，返回结果字典"""
    script_path = os.path.join(BASE_DIR, script_rel)
    result = subprocess.run(
        ["python", script_path], capture_output=True, text=True, timeout=1800, **kwargs
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script_rel} 失败 (code={result.returncode}): {result.stderr[-500:]}")
    return {"returncode": result.returncode, "stdout": result.stdout[-500:]}


@shared_task(name="app.tasks.data_tasks.run_clustering_task")
def run_clustering_task():
    """执行聚类分析脚本。替代 scheduler.py 中的 run_clustering_task。"""
    logger.info("开始执行定时聚类任务...")
    return _run_script("scripts/topic_clustering.py")


@shared_task(name="app.tasks.data_tasks.crawl_task")
def crawl_task(incremental: bool = True, target_count: int = 500):
    """增量爬取 B站数据"""
    scrapy_dir = os.path.join(BASE_DIR, "src", "bilibili_scraper")
    result = subprocess.run(
        [
            "scrapy", "crawl", "video",
            f"-a", f"incremental={incremental}",
            f"-a", f"target_count={target_count}",
        ],
        capture_output=True, text=True, timeout=3600, cwd=scrapy_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(f"爬虫失败 (code={result.returncode}): {result.stderr[-500:]}")
    return {"returncode": result.returncode, "stdout": result.stdout[-500:]}


@shared_task(name="app.tasks.data_tasks.load_db_task")
def load_db_task():
    """清洗数据并入库"""
    return _run_script("scripts/load_data_to_db.py")


@shared_task(name="app.tasks.data_tasks.train_task")
def train_task():
    """重新训练预测模型"""
    return _run_script("scripts/train_view_predictor.py")

