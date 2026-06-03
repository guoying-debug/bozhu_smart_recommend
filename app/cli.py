"""
app/cli.py —— 统一命令行入口
用法:
    python -m app.cli serve
    python -m app.cli worker
    python -m app.cli beat
    python -m app.cli pipeline [--no-crawl] [--no-train] [--no-cluster]
    python -m app.cli migrate
"""
import os
import sys
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_serve(args):
    """启动 Flask/Gunicorn Web 服务"""
    workers = getattr(args, "workers", 1)
    os.execv(sys.executable, [sys.executable, "-m", "gunicorn",
                               "-w", str(workers), "-b", "0.0.0.0:5000", "app:create_app()"])


def cmd_worker(args):
    """启动 Celery Worker"""
    concurrency = getattr(args, "concurrency", 2)
    os.execv(sys.executable, [sys.executable, "-m", "celery",
                               "-A", "app.celery_app", "worker",
                               "--loglevel=info", "-c", str(concurrency)])


def cmd_beat(args):
    """启动 Celery Beat 调度器"""
    os.execv(sys.executable, [sys.executable, "-m", "celery",
                               "-A", "app.celery_app", "beat", "--loglevel=info"])


def cmd_pipeline(args):
    """触发完整数据 pipeline（爬取→入库→训练→聚类）"""
    from startup import run_pipeline
    run_pipeline(
        crawl=not getattr(args, "no_crawl", False),
        train=not getattr(args, "no_train", False),
        cluster=not getattr(args, "no_cluster", False),
    )


def cmd_migrate(args):
    """运行 Alembic 数据库迁移到最新版本"""
    result = subprocess.run(["alembic", "upgrade", "head"], cwd=BASE_DIR)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="博主智策 统一CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="启动 Web 服务")
    p_serve.add_argument("-w", "--workers", type=int, default=1)

    p_worker = sub.add_parser("worker", help="启动 Celery Worker")
    p_worker.add_argument("-c", "--concurrency", type=int, default=2)

    sub.add_parser("beat", help="启动 Celery Beat")

    p_pipeline = sub.add_parser("pipeline", help="触发数据 pipeline")
    p_pipeline.add_argument("--no-crawl", action="store_true")
    p_pipeline.add_argument("--no-train", action="store_true")
    p_pipeline.add_argument("--no-cluster", action="store_true")

    sub.add_parser("migrate", help="运行 Alembic 数据库迁移")

    args = parser.parse_args()
    {"serve": cmd_serve, "worker": cmd_worker, "beat": cmd_beat,
     "pipeline": cmd_pipeline, "migrate": cmd_migrate}[args.command](args)


if __name__ == "__main__":
    main()
