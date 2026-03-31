"""
测试 predictor 模块（mock 模型，验证返回格式）。
"""
import pytest
from unittest.mock import MagicMock, patch
from app.core.predictor import predict_view, predict_bucket
from app.models.schemas import PredictRequest


@patch('app.core.predictor._VIEW_PREDICTOR')
def test_predict_view_returns_tuple(mock_predictor):
    """测试 predict_view 返回 4 元组"""
    mock_model = MagicMock()
    mock_model.predict.return_value = [10.0]  # log 值
    mock_predictor.__getitem__.side_effect = lambda k: {
        "model": mock_model,
        "artifact": {"numeric_columns": ["title_len"]}
    }[k]

    req = PredictRequest(title="测试标题")
    pred_view, bucket_id, bucket_name, explanations = predict_view(req)

    assert isinstance(pred_view, float)
    assert isinstance(bucket_id, int)
    assert isinstance(bucket_name, str)
    assert isinstance(explanations, list)


def test_predict_view_raises_when_model_not_loaded():
    """测试模型未加载时抛出 RuntimeError"""
    with patch('app.core.predictor._VIEW_PREDICTOR', None):
        req = PredictRequest(title="测试")
        with pytest.raises(RuntimeError, match="播放量预测模型未加载"):
            predict_view(req)
