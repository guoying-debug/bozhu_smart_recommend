"""
数据库引擎与会话管理
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from app.core.config import get_database_url

# 创建引擎（全局单例）
DATABASE_URL = get_database_url()
if not DATABASE_URL:
    raise RuntimeError(
        "数据库连接 URL 未配置。请设置 DB_PASSWORD 环境变量。"
    )

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_session():
    """
    上下文管理器，用于自动管理会话生命周期

    用法:
        with get_session() as session:
            session.query(Video).all()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
