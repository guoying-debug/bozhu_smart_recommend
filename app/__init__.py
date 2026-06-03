import os

from flask import Flask, render_template
from app.api.routes import api_bp
from app.core.data_loader import load_data
from app.core.predictor import load_models
from app.core.recommender import init_recommender

from prometheus_flask_exporter import PrometheusMetrics

def create_app():
    app = Flask(__name__)

    metrics = PrometheusMetrics(app)
    metrics.info('app_info', 'Application info', version='1.0.3')

    app.register_blueprint(api_bp)

    @app.route('/')
    def index():
        return render_template('index.html')

    with app.app_context():
        print("正在初始化应用...")
        load_data()
        load_models()
        init_recommender()
        print("应用初始化完成。调度任务由 Celery Beat 负责，已移除 APScheduler。")

    return app
