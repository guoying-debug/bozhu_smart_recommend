"""
测试 feature_utils 中的特征工程函数。
"""
import pytest
from app.utils.feature_utils import parse_publish_time, title_engineered_features


def test_parse_publish_time_valid():
    """测试合法的时间戳解析"""
    import pandas as pd
    ts = 1700000000  # 2023-11-14
    dt = parse_publish_time(ts)
    assert not pd.isna(dt)
    assert dt.year == 2023


def test_parse_publish_time_invalid():
    """测试非法时间戳返回 NaT"""
    import pandas as pd
    dt = parse_publish_time("invalid")
    assert pd.isna(dt)


def test_title_engineered_features():
    """测试标题特征提取"""
    features = title_engineered_features("【Python教程】3天速成？")
    assert features["title_len"] == 14
    assert features["question_cnt"] == 1
    assert features["has_brackets"] == 1
    assert features["has_number"] == 1
