from apscheduler.schedulers.background import BackgroundScheduler
import time
import atexit
import logging
import subprocess
import os

# 不在模块顶层调用 logging.basicConfig，避免覆盖应用工厂已配置的日志格式。
# 日志配置统一由 app/__init__.py（应用工厂）负责。
logger = logging.getLogger(__name__)

def run_clustering_task():
    """
    执行聚类分析脚本的任务。
    """
    logger.info("开始执行定时聚类任务...")
    
    # 获取脚本路径
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_path = os.path.join(base_dir, 'scripts', 'topic_clustering.py')
    
    try:
        # 使用 subprocess 运行脚本，确保环境隔离且不阻塞主线程
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("定时聚类任务执行成功！")
            logger.info(result.stdout[-200:]) # 打印最后200字符的输出
        else:
            logger.error(f"定时聚类任务失败，返回码: {result.returncode}")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"执行聚类脚本时发生异常: {e}")

def init_scheduler(app):
    """
    初始化调度器。
    """
    scheduler = BackgroundScheduler()
    
    # 添加每日凌晨 2 点执行的任务
    scheduler.add_job(func=run_clustering_task, trigger="cron", hour=2, minute=0)
    
    # 为了演示，也可以添加一个每隔 12 小时执行的任务
    # scheduler.add_job(func=run_clustering_task, trigger="interval", hours=12)
    
    scheduler.start()
    logger.info("定时任务调度器已启动。")
    
    # 注册退出时的清理函数
    atexit.register(lambda: scheduler.shutdown())
