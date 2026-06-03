"""
Video 表数据访问层（Repository 模式）
"""
from typing import List, Optional, Dict, Any
from sqlalchemy import text, func
from sqlalchemy.dialects.mysql import insert
from app.db.session import get_session
from app.db.models.video import Video


class VideoRepository:
    """Video 表的所有读写操作集中在此"""

    @staticmethod
    def bulk_upsert(records: List[Dict[str, Any]]) -> int:
        """
        批量插入或更新（MySQL INSERT ... ON DUPLICATE KEY UPDATE）

        替代原 load_data_to_db.py 中的 TRUNCATE + bulk_insert，
        支持增量更新而不丢失历史数据。

        Args:
            records: 字典列表，每个字典对应一条 Video 记录

        Returns:
            插入或更新的记录数
        """
        if not records:
            return 0

        with get_session() as session:
            # 构造 INSERT ... ON DUPLICATE KEY UPDATE
            stmt = insert(Video).values(records)
            # 重复时更新所有字段（除主键外）
            update_cols = {
                col.name: stmt.inserted[col.name]
                for col in Video.__table__.columns
                if col.name != "video_id"
            }
            upsert_stmt = stmt.on_duplicate_key_update(**update_cols)

            result = session.execute(upsert_stmt)
            session.commit()
            return result.rowcount

    @staticmethod
    def truncate_and_insert(records: List[Dict[str, Any]]) -> int:
        """
        清空表后批量插入（兼容旧逻辑）

        Args:
            records: 字典列表

        Returns:
            插入的记录数
        """
        with get_session() as session:
            session.execute(text("TRUNCATE TABLE videos;"))
            session.bulk_insert_mappings(Video, records)
            session.commit()
            return len(records)

    @staticmethod
    def get_categories() -> List[str]:
        """
        获取所有不重复的分类列表

        替代 data_loader.py 中的裸 SQL
        """
        with get_session() as session:
            result = session.query(Video.category).distinct().all()
            return [row[0] for row in result if row[0]]

    @staticmethod
    def get_video_category_mapping() -> Dict[str, str]:
        """
        返回 {video_id: category} 映射

        用于 data_loader 补全 CSV 中缺失的 category 列
        """
        with get_session() as session:
            result = session.query(Video.video_id, Video.category).all()
            return {row[0]: row[1] for row in result}

    @staticmethod
    def count() -> int:
        """返回 videos 表总记录数"""
        with get_session() as session:
            return session.query(func.count(Video.video_id)).scalar()

    @staticmethod
    def get_by_id(video_id: str) -> Optional[Video]:
        """根据 video_id 查询单条记录"""
        with get_session() as session:
            return session.query(Video).filter(Video.video_id == video_id).first()

    @staticmethod
    def get_all(limit: int = 1000) -> List[Video]:
        """
        查询所有记录（带分页保护）

        Args:
            limit: 最大返回条数
        """
        with get_session() as session:
            return session.query(Video).limit(limit).all()
