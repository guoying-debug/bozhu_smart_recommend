import os
from dotenv import load_dotenv

# --- 基础路径配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 数据目录
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
CHROMA_DB_DIR = os.path.join(BASE_DIR, 'chroma_db')

# 模型目录
MODELS_TRAINED_DIR = os.path.join(BASE_DIR, 'models', 'trained')

# 静态资源目录
STATIC_DIR = os.path.join(BASE_DIR, 'app', 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')

# 具体文件路径
DATA_PATH = os.path.join(DATA_PROCESSED_DIR, 'videos_with_clusters_and_coords.csv')
VIEW_PREDICTOR_PATH = os.path.join(MODELS_TRAINED_DIR, 'view_predictor.joblib')
VIEW_BUCKET_CLASSIFIER_PATH = os.path.join(MODELS_TRAINED_DIR, 'view_bucket_classifier.joblib')
BERT_EMBEDDINGS_PATH = os.path.join(MODELS_TRAINED_DIR, 'bert_embeddings.npy')
BERT_KMEANS_MODEL_PATH = os.path.join(MODELS_TRAINED_DIR, 'bert_kmeans_model.pkl')
BERT_DBSCAN_MODEL_PATH = os.path.join(MODELS_TRAINED_DIR, 'bert_dbscan_model.pkl')

# 数据库配置
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "bilibili_data")

# LLM 配置
LLM_API_KEY = (
    os.getenv("ZHIPU_API_KEY")
    or os.getenv("ZHIPUAI_API_KEY")
    or os.getenv("LLM_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash-250414")

def get_database_url():
    if not DB_PASSWORD:
        return None
    return f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
