"""
测试 Flask API routes（mock 核心服务，验证 HTTP 响应）。
"""
import pytest
from unittest.mock import patch, MagicMock
from app import create_app


@pytest.fixture
def client():
    """创建测试客户端"""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@patch('app.api.routes.predict_view')
def test_api_predict_view_success(mock_predict, client):
    """测试 /api/predict_view 成功返回"""
    mock_predict.return_value = (5000.0, 2, "中等", [{"feature": "标题长度", "effect": "正向", "reason": "适中"}])

    response = client.post('/api/predict_view', json={"title": "测试标题"})
    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_view" in data
    assert data["predicted_bucket"] == "中等"


def test_api_predict_view_missing_title(client):
    """测试缺少 title 字段时返回 400"""
    response = client.post('/api/predict_view', json={})
    assert response.status_code == 400


@patch('app.api.routes.get_similar_titles')
@patch('app.api.routes.predict_view')
@patch('app.api.routes.analyze_title_with_llm')
def test_api_title_advice(mock_llm, mock_predict, mock_similar, client):
    """测试 /api/title_advice 返回建议"""
    mock_similar.return_value = [{"title": "爆款", "view_count": 100000, "similarity": 0.9, "cluster": 1, "category": "知识", "sources": ["dense"]}]
    mock_predict.return_value = (5000.0, 2, "中等", [])
    mock_llm.return_value = {"suggestions": ["建议1"], "diagnosis": "诊断"}

    response = client.post('/api/title_advice', json={"title": "测试"})
    assert response.status_code == 200
    data = response.get_json()
    assert "similar_titles" in data
    assert "advice_list" in data
