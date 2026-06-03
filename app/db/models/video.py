"""
Video model —— 从 scripts/db_setup.py 迁移而来
"""
from sqlalchemy import Column, Integer, String, BigInteger, DateTime, JSON, Text
from app.db.base import Base


class Video(Base):
    __tablename__ = "videos"

    video_id = Column(String(32), primary_key=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    author = Column(String(255))
    author_id = Column(BigInteger)
    publish_time = Column(DateTime, nullable=False)
    view_count = Column(Integer)
    like_count = Column(Integer)
    coin_count = Column(Integer)
    favorite_count = Column(Integer)
    share_count = Column(Integer)
    comment_count = Column(Integer)
    tags = Column(JSON)
    category = Column(String(100))

    def __repr__(self):
        return f"<Video(title='{self.title[:30]}...')>"
