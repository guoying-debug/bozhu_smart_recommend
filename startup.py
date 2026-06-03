"""
startup.py —— 一键触发完整数据 pipeline（爬取→入库→训练→聚类）
使用 Celery chain，每步独立可重试，失败不影响已完成的步骤。
"""
import os
import sys
import logging
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _require_env():
    """确保关键环境变量已设置"""
    import getpass
    if not (os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY") or os.getenv("LLM_API_KEY")):
        key = input("请输入智谱 API Key（留空跳过）: ").strip()
        if key:
            os.environ["ZHIPU_API_KEY"] = key
    if not os.getenv("DB_PASSWORD"):
        pwd = getpass.getpass("请输入 MySQL 数据库密码: ").strip()
        if pwd:
            os.environ["DB_PASSWORD"] = pwd


def run_pipeline(crawl: bool = True, train: bool = True, cluster: bool = True):
    """
    通过 Celery chain 串行执行完整数据 pipeline。
    各步骤使用独立 Celery task，失败后可单独重跑。
    """
    from celery import chain
    from app.tasks.data_tasks import crawl_task, load_db_task, train_task, run_clustering_task

    steps = []
    if crawl:
        steps.append(crawl_task.si())
    steps.append(load_db_task.si())
    if train:
        steps.append(train_task.si())
    if cluster:
        steps.append(run_clustering_task.si())

    if not steps:
        logger.warning("没有选择任何步骤，退出")
        return

    logger.info("提交 pipeline，步骤数: %d", len(steps))
    result = chain(*steps).apply_async()
    logger.info("Pipeline 已提交，task chain id: %s", result.id)
    logger.info("可通过 GET /api/task/<id> 或 Flower 监控进度")
    return result


def main():
    _require_env()
    import yaml
    config_path = os.path.join(BASE_DIR, "startup_config.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    run_pipeline(
        crawl=config.get("ENABLE_CRAWL", True),
        train=config.get("RETRAIN_MODEL", True),
        cluster=config.get("REGENERATE_CLUSTERING", True),
    )


if __name__ == "__main__":
    main()
