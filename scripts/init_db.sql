-- 初始化脚本由 docker-compose 在 db 容器首次启动时自动执行
-- 对应 db_setup.py 中 SQLAlchemy ORM 定义的 videos 表结构
CREATE DATABASE IF NOT EXISTS bilibili_data CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE bilibili_data;

CREATE TABLE IF NOT EXISTS videos (
    video_id    VARCHAR(32)   NOT NULL PRIMARY KEY,
    title       TEXT          NOT NULL,
    description TEXT,
    author      VARCHAR(255),
    author_id   BIGINT,
    publish_time DATETIME     NOT NULL,
    view_count  INT,
    like_count  INT,
    coin_count  INT,
    favorite_count INT,
    share_count INT,
    comment_count  INT,
    tags        JSON,
    category    VARCHAR(100)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
