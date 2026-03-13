import re
import pandas as pd
import numpy as np

# 扩充情绪词库
EMOTION_WORDS = [
    # 强情绪 (High Arousal)
    "震惊", "惊呆", "惊了", "离谱", "离大谱", "绝了", "太强", "逆天", "封神", "上头",
    "笑死", "爆笑", "泪目", "爆哭", "崩溃", "翻车", "血亏", "后悔", "破防", "治愈", "感动",
    "必须", "千万", "绝对", "居然", "竟然", "竟然", "卧槽", "牛逼", "炸裂", "恐怖",
    # 中情绪 / 正向 (Positive)
    "开心", "快乐", "喜欢", "推荐", "好用", "神器", "福利", "惊喜", "完美", "首发",
    "独家", "揭秘", "真相", "干货", "良心", "免费", "白嫖", "赚了", "超值",
    # 中情绪 / 负向 (Negative - 用于引起共鸣)
    "难过", "失望", "避雷", "踩雷", "劝退", "恶心", "无语", "生气", "痛苦", "惨案"
]

def parse_publish_time(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(int(value), unit="s", errors="coerce")
    return pd.to_datetime(value, errors="coerce")

def title_engineered_features(title: str):
    title = "" if pd.isna(title) else str(title)
    emotion_cnt = 0
    emotion_hit_cnt = 0
    for w in EMOTION_WORDS:
        c = title.count(w)
        if c > 0:
            emotion_hit_cnt += 1
            emotion_cnt += c
    return {
        "title_len": len(title),
        "digit_cnt": len(re.findall(r"\d", title)),
        "question_cnt": title.count("?") + title.count("？"),
        "exclam_cnt": title.count("!") + title.count("！"),
        "has_brackets": 1 if any(ch in title for ch in ["【", "】", "[", "]", "（", "）", "(", ")", "《", "》"]) else 0,
        "has_percent": 1 if "%" in title else 0,
        "has_colon": 1 if any(ch in title for ch in [":", "："]) else 0,
        "has_tutorial": 1 if any(k in title for k in ["教程", "教学", "入门", "新手", "一图流", "指南"]) else 0,
        "has_review": 1 if any(k in title for k in ["测评", "评测", "对比", "开箱"]) else 0,
        "has_list": 1 if any(k in title for k in ["盘点", "合集", "汇总", "TOP", "Top", "top"]) else 0,
        "has_hot": 1 if any(k in title for k in ["爆", "最强", "必看", "热门", "全网", "火爆"]) else 0,
        "emotion_cnt": emotion_cnt,
        "emotion_hit_cnt": emotion_hit_cnt,
    }

def make_bucket(view_count: float):
    if view_count < 100_000:
        return 0
    if view_count < 500_000:
        return 1
    if view_count < 2_000_000:
        return 2
    return 3

def bucket_name(bucket_id: int):
    return {0: "0-10万", 1: "10-50万", 2: "50-200万", 3: "200万+"}.get(int(bucket_id), "未知")
