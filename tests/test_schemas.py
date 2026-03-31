"""
测试 Pydantic schemas 的字段校验逻辑。
"""
import pytest
from pydantic import ValidationError
from app.models.schemas import PredictRequest, PredictViewResponse


def test_predict_request_valid():
    """测试合法的 PredictRequest"""
    req = PredictRequest(title="测试标题", category="知识")
    assert req.title == "测试标题"
    assert req.category == "知识"


def test_predict_request_missing_title():
    """测试缺少必填字段 title 时抛出 ValidationError"""
    with pytest.raises(ValidationError):
        PredictRequest(category="知识")


def test_predict_request_default_category():
    """测试 category 有默认值 '未知'"""
    req = PredictRequest(title="测试")
    assert req.category == "未知"


def test_predict_view_response():
    """测试 PredictViewResponse 结构"""
    resp = PredictViewResponse(predicted_view=5000.0, predicted_bucket="中等", bucket_id=2)
    assert resp.predicted_view == 5000.0
    assert resp.bucket_id == 2
