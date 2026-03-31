"""
项目启动脚本：自动抓取最新数据并初始化所有模块
"""
import os
import sys
import subprocess
import logging
from datetime import datetime
import yaml
import getpass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_environment():
    """交互式设置环境变量"""
    print("\n" + "="*60)
    print("环境配置")
    print("="*60)

    # 检查并设置 API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        api_key = input("请输入阿里云 DashScope API Key: ").strip()
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
        else:
            logger.warning("未设置 API Key，LLM 功能将不可用")

    # 检查并设置数据库密码
    if not os.getenv("DB_PASSWORD"):
        db_password = getpass.getpass("请输入 MySQL 数据库密码: ").strip()
        if db_password:
            os.environ["DB_PASSWORD"] = db_password
        else:
            logger.warning("未设置数据库密码，数据库相关功能将不可用")

    print("="*60 + "\n")

def load_config():
    """加载配置文件"""
    config_path = os.path.join(BASE_DIR, "startup_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {
        'ENABLE_CRAWL': True,
        'CRAWL_TARGET_COUNT': 500,
        'CRAWL_INCREMENTAL': True,
        'RETRAIN_MODEL': True,
        'REGENERATE_CLUSTERING': True,
        'CONTINUE_ON_ERROR': True
    }

def run_command(cmd, cwd=None, description=""):
    """执行命令并实时输出"""
    logger.info(f"{'='*60}")
    logger.info(f"开始执行: {description}")
    logger.info(f"命令: {cmd}")
    logger.info(f"{'='*60}")

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd or BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode == 0:
            logger.info(f"✅ {description} - 完成")
            return True
        else:
            logger.error(f"❌ {description} - 失败 (返回码: {process.returncode})")
            return False
    except Exception as e:
        logger.error(f"❌ {description} - 异常: {e}")
        return False

def main():
    # 交互式设置环境变量
    setup_environment()

    config = load_config()

    logger.info(f"\n{'#'*60}")
    logger.info(f"# 博主智策 - 自动化启动流程")
    logger.info(f"# 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*60}\n")

    # 步骤1: 增量抓取最新数据
    if config.get('ENABLE_CRAWL', True):
        logger.info("📡 步骤 1/5: 抓取最新 B站数据...")
        scrapy_dir = os.path.join(BASE_DIR, "src", "bilibili_scraper")
        incremental = "True" if config.get('CRAWL_INCREMENTAL', True) else "False"
        target = config.get('CRAWL_TARGET_COUNT', 500)

        success = run_command(
            f"scrapy crawl video -a incremental={incremental} -a target_count={target}",
            cwd=scrapy_dir,
            description="增量爬虫"
        )

        if not success and not config.get('CONTINUE_ON_ERROR', True):
            logger.error("爬虫失败，终止启动流程")
            return
    else:
        logger.info("⏭️ 跳过数据抓取（配置已禁用）")

    # 步骤2: 数据入库
    logger.info("\n💾 步骤 2/5: 数据清洗与入库...")
    run_command("python scripts/load_data_to_db.py", description="数据入库")

    # 步骤3: 重新训练预测模型
    if config.get('RETRAIN_MODEL', True):
        logger.info("\n🤖 步骤 3/5: 训练播放量预测模型...")
        run_command("python scripts/train_view_predictor.py", description="模型训练")
    else:
        logger.info("⏭️ 跳过模型训练（配置已禁用）")

    # 步骤4: 重新生成聚类和向量库
    if config.get('REGENERATE_CLUSTERING', True):
        logger.info("\n🔍 步骤 4/5: 生成话题聚类与向量库...")
        run_command("python scripts/topic_clustering.py", description="聚类分析")
    else:
        logger.info("⏭️ 跳过聚类生成（配置已禁用）")

    # 步骤5: 启动 Web 服务
    logger.info("\n🚀 步骤 5/5: 启动 Flask Web 服务...")
    logger.info("服务地址: http://127.0.0.1:5000")
    logger.info("按 Ctrl+C 停止服务\n")
    run_command("python app.py", description="Web 服务")

if __name__ == "__main__":
    main()
