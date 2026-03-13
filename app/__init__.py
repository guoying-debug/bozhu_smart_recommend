from flask import Flask, render_template
from app.api.routes import api_bp
from app.core.data_loader import load_data
from app.core.predictor import load_models
from app.core.recommender import init_recommender
from app.core.scheduler import init_scheduler

from prometheus_flask_exporter import PrometheusMetrics

def create_app():
    app = Flask(__name__)
    
    # 初始化 Prometheus 监控
    # 这会自动暴露 /metrics 端点，并记录请求延迟、状态码等
    metrics = PrometheusMetrics(app)
    metrics.info('app_info', 'Application info', version='1.0.3')
    
    # 注册蓝图
    app.register_blueprint(api_bp)
    
    # 首页路由
    @app.route('/')
    def index():
        return render_template('index.html')
        
    # 初始化数据和模型
    with app.app_context():
        print("正在初始化应用...")
        load_data()
        load_models()
        init_recommender()
        # 启动定时任务
        try:
            init_scheduler(app)
        except Exception as e:
            print(f"定时任务启动失败 (可能在调试模式下重复启动): {e}")
            
        print("应用初始化完成。")
        
    return app
