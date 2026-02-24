from flask import Flask, jsonify, render_template, request
import joblib
import os
import pandas as pd
from sqlalchemy import create_engine
import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 配置部分 ---
# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据和模型存放目录
DATA_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# CSV数据文件路径
DATA_PATH = os.path.join(DATA_DIR, 'videos_with_clusters_and_coords.csv')
VIEW_PREDICTOR_PATH = os.path.join(DATA_DIR, 'view_predictor.joblib')
VIEW_BUCKET_CLASSIFIER_PATH = os.path.join(DATA_DIR, 'view_bucket_classifier.joblib')

# --- 数据加载部分 ---
df = None
cluster_summary = {}
view_predictor = None
view_bucket_classifier = None
similarity_vectorizer = None


similarity_matrix = None
category_stats = None

def _jieba_tokenize(text: str):
    return jieba.lcut(text)

EMOTION_WORDS = [
    "震惊",
    "惊呆",
    "惊了",
    "离谱",
    "离大谱",
    "绝了",
    "太强",
    "逆天",
    "封神",
    "上头",
    "笑死",
    "爆笑",
    "泪目",
    "爆哭",
    "崩溃",
    "翻车",
    "血亏",
    "后悔",
    "破防",
    "治愈",
    "感动",
]

def _get_database_url():
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "bilibili_data")
    if not db_password:
        return None
    return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def _get_stopwords():
    return {
        "的", "是", "了", "在", "我", "你", "他", "她", "它", "们", "都", "也",
        "一个", "怎么", "这种", "这个", "一下", "真的", "就是", "还是", "不是",
        "什么", "为什么", "可以", "没有", "以及", "但是", "因为", "如果", "所以",
    }

def _top_keywords_from_titles(titles, top_k=10):
    stopwords = _get_stopwords()
    counts = {}
    for t in titles:
        t = "" if pd.isna(t) else str(t)
        t = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", " ", t).strip()
        for w in jieba.lcut(t):
            w = w.strip()
            if not w:
                continue
            if w in stopwords:
                continue
            if len(w) <= 1:
                continue
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]

def _parse_publish_time(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (int, np.integer, float, np.floating)):
        return pd.to_datetime(int(value), unit="s", errors="coerce")
    return pd.to_datetime(value, errors="coerce")

def _title_engineered_features(title: str):
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

def _make_bucket(view_count: float):
    if view_count < 100_000:
        return 0
    if view_count < 500_000:
        return 1
    if view_count < 2_000_000:
        return 2
    return 3

def _bucket_name(bucket_id: int):
    return {0: "0-10万", 1: "10-50万", 2: "50-200万", 3: "200万+"}.get(int(bucket_id), "未知")

def _build_model_input(artifact: dict, title: str, category: str, author_id=None, publish_time=None):
    numeric_cols = artifact.get("numeric_columns")
    if not numeric_cols:
        return pd.DataFrame([{"title": title, "category": category}])

    row = {"title": title, "category": category}
    dt = _parse_publish_time(publish_time)
    row["hour"] = int(dt.hour) if not pd.isna(dt) else -1
    row["dow"] = int(dt.dayofweek) if not pd.isna(dt) else -1

    author_stats = artifact.get("author_stats") or {}
    author_defaults = artifact.get("author_defaults") or {
        "author_video_count": 0.0,
        "author_mean_view": 0.0,
        "author_median_view": 0.0,
    }

    aid = pd.to_numeric(author_id, errors="coerce")
    if pd.isna(aid):
        author_feat = author_defaults
    else:
        author_feat = author_stats.get(int(aid), author_defaults)
    row.update(author_feat)

    row.update(_title_engineered_features(title))
    for c in numeric_cols:
        row[c] = float(pd.to_numeric(row.get(c, 0.0), errors="coerce") if row.get(c, 0.0) is not None else 0.0)
        if np.isnan(row[c]):
            row[c] = 0.0

    return pd.DataFrame([row])

def load_data():
    """
    从CSV文件加载预处理和聚类过的数据。
    """
    global df, cluster_summary, view_predictor, view_bucket_classifier, similarity_vectorizer, similarity_matrix, category_stats
    
    print("正在加载预处理数据...")
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"成功从 {DATA_PATH} 加载了 {len(df)} 条记录。")
    else:
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}。请先运行 topic_clustering.py 脚本。")

    database_url = _get_database_url()
    if database_url and "category" not in df.columns:
        try:
            engine = create_engine(database_url)
            cat_df = pd.read_sql("SELECT video_id, category FROM videos", engine)
            df = df.merge(cat_df, on="video_id", how="left")
            df["category"] = df["category"].fillna("未知").astype(str)
        except Exception:
            pass

    # --- 生成聚类摘要 ---
    # K-Means摘要
    kmeans_counts = df['bert_kmeans_cluster'].value_counts().to_dict()
    cluster_summary['kmeans'] = {int(k): int(v) for k, v in kmeans_counts.items()}
    
    # DBSCAN摘要
    dbscan_counts = df['dbscan_cluster'].value_counts().to_dict()
    n_clusters_dbscan = len(set(df['dbscan_cluster'])) - (1 if -1 in set(df['dbscan_cluster']) else 0)
    n_noise = dbscan_counts.get(-1, 0)
    
    cluster_summary['dbscan'] = {
        'total_clusters': n_clusters_dbscan,
        'noise_points': int(n_noise),
        'noise_percentage': round(n_noise / len(df) * 100, 2),
        'cluster_counts': {int(k): int(v) for k, v in dbscan_counts.items() if k != -1}
    }

    if "category" in df.columns:
        category_grouped = df.groupby("category")["view_count"]
        category_stats = (
            category_grouped.agg(["count", "mean", "median"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )

    if len(df) > 0:
        similarity_vectorizer = TfidfVectorizer(
            tokenizer=_jieba_tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
        )
        similarity_matrix = similarity_vectorizer.fit_transform(df["title"].fillna("").astype(str))

    if os.path.exists(VIEW_PREDICTOR_PATH):
        try:
            view_predictor = joblib.load(VIEW_PREDICTOR_PATH)
        except Exception:
            view_predictor = None
    if os.path.exists(VIEW_BUCKET_CLASSIFIER_PATH):
        try:
            view_bucket_classifier = joblib.load(VIEW_BUCKET_CLASSIFIER_PATH)
        except Exception:
            view_bucket_classifier = None

    print("数据加载和摘要生成完成。")


# --- Flask应用部分 ---
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    """
    提供主HTML页面。
    """
    return render_template('index.html')

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """
    返回K-Means聚类后的话题摘要信息。
    为了与旧版前端兼容，我们模拟旧的数据结构。
    """
    if df is None:
        return jsonify({"error": "数据尚未初始化"}), 500
    
    # 按K-Means聚类分组，并计算统计信息
    grouped = df.groupby('bert_kmeans_cluster')
    
    results = []
    for cluster_id, group in grouped:
        top_terms = _top_keywords_from_titles(group["title"].head(200).tolist(), top_k=10)
        
        results.append({
            'cluster_id': int(cluster_id),
            'top_terms': top_terms,
            'total_videos': len(group),
            'total_views': int(group['view_count'].sum()),
            'average_views': float(group['view_count'].mean()),
            'top_videos': group.sort_values('view_count', ascending=False).head(5)[['title', 'view_count']].to_dict('records')
        })
        
    sorted_results = sorted(results, key=lambda x: x['total_views'], reverse=True)
    return jsonify(sorted_results)

@app.route('/api/analysis_summary', methods=['GET'])
def get_analysis_summary():
    """
    返回新的聚类分析摘要。
    """
    if not cluster_summary:
        return jsonify({"error": "分析摘要尚未生成"}), 500
    return jsonify(cluster_summary)

@app.route('/api/predict_view', methods=['POST'])
def predict_view():
    if view_predictor is None:
        return jsonify({"error": "预测模型未加载，请先运行 scripts/train_view_predictor.py"}), 500

    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    category = (payload.get("category") or "未知").strip() or "未知"
    author_id = payload.get("author_id")
    publish_time = payload.get("publish_time")
    if not title:
        return jsonify({"error": "title 不能为空"}), 400

    pipeline = view_predictor.get("pipeline")
    if pipeline is None:
        return jsonify({"error": "预测模型格式不正确"}), 500

    X = _build_model_input(view_predictor, title, category, author_id=author_id, publish_time=publish_time)
    y_pred = float(pipeline.predict(X)[0])
    view_pred = float(np.expm1(y_pred))
    view_pred = int(max(0.0, round(view_pred)))

    bucket_pred = _make_bucket(view_pred)
    bucket_from_classifier = None
    bucket_probs = None
    if view_bucket_classifier is not None and view_bucket_classifier.get("pipeline") is not None:
        cls_X = _build_model_input(view_bucket_classifier, title, category, author_id=author_id, publish_time=publish_time)
        cls_pipeline = view_bucket_classifier["pipeline"]
        try:
            bucket_from_classifier = int(cls_pipeline.predict(cls_X)[0])
            if hasattr(cls_pipeline, "predict_proba"):
                probs = cls_pipeline.predict_proba(cls_X)[0].tolist()
                names = (view_bucket_classifier.get("metrics") or {}).get("label_names") or ["0-10万", "10-50万", "50-200万", "200万+"]
                bucket_probs = {names[i]: float(probs[i]) for i in range(min(len(names), len(probs)))}
        except Exception:
            bucket_from_classifier = None
            bucket_probs = None

    return jsonify(
        {
            "title": title,
            "category": category,
            "author_id": author_id,
            "publish_time": publish_time,
            "predicted_view_count": view_pred,
            "predicted_log1p_view_count": y_pred,
            "predicted_bucket_id": int(bucket_from_classifier if bucket_from_classifier is not None else bucket_pred),
            "predicted_bucket_name": _bucket_name(int(bucket_from_classifier if bucket_from_classifier is not None else bucket_pred)),
            "bucket_probabilities": bucket_probs,
            "model_metrics": view_predictor.get("metrics", {}),
        }
    )

@app.route('/api/predict_bucket', methods=['POST'])
def predict_bucket():
    if view_bucket_classifier is None or view_bucket_classifier.get("pipeline") is None:
        return jsonify({"error": "分档模型未加载，请先运行 scripts/train_view_predictor.py"}), 500

    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    category = (payload.get("category") or "未知").strip() or "未知"
    author_id = payload.get("author_id")
    publish_time = payload.get("publish_time")
    if not title:
        return jsonify({"error": "title 不能为空"}), 400

    pipeline = view_bucket_classifier["pipeline"]
    X = _build_model_input(view_bucket_classifier, title, category, author_id=author_id, publish_time=publish_time)
    bucket_id = int(pipeline.predict(X)[0])
    result = {
        "title": title,
        "category": category,
        "author_id": author_id,
        "publish_time": publish_time,
        "predicted_bucket_id": bucket_id,
        "predicted_bucket_name": _bucket_name(bucket_id),
        "model_metrics": view_bucket_classifier.get("metrics", {}),
    }
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[0].tolist()
        names = (view_bucket_classifier.get("metrics") or {}).get("label_names") or ["0-10万", "10-50万", "50-200万", "200万+"]
        result["bucket_probabilities"] = {names[i]: float(probs[i]) for i in range(min(len(names), len(probs)))}
    return jsonify(result)

@app.route('/api/title_advice', methods=['POST'])
def title_advice():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    category = (payload.get("category") or "未知").strip() or "未知"
    author_id = payload.get("author_id")
    publish_time = payload.get("publish_time")
    if not title:
        return jsonify({"error": "title 不能为空"}), 400

    predicted = None
    predicted_bucket = None
    if view_predictor is not None and view_predictor.get("pipeline") is not None:
        X = _build_model_input(view_predictor, title, category, author_id=author_id, publish_time=publish_time)
        y_pred = float(view_predictor["pipeline"].predict(X)[0])
        predicted = int(max(0.0, round(float(np.expm1(y_pred)))))
        predicted_bucket = _bucket_name(_make_bucket(predicted))

    similar = []
    keywords = []
    if df is not None and similarity_vectorizer is not None and similarity_matrix is not None:
        q = similarity_vectorizer.transform([title])
        scores = (similarity_matrix @ q.T).toarray().ravel()
        top_idx = np.argsort(-scores)[:10]
        rows = df.iloc[top_idx].copy()
        rows["score"] = scores[top_idx]
        cols = ["video_id", "title", "view_count"]
        if "category" in rows.columns:
            cols.append("category")
        for _, r in rows.iterrows():
            item = {
                "video_id": r.get("video_id"),
                "title": r.get("title"),
                "view_count": int(r.get("view_count", 0)),
                "similarity": float(r.get("score", 0.0)),
            }
            if "category" in rows.columns:
                item["category"] = r.get("category")
            similar.append(item)

        keywords = _top_keywords_from_titles([s["title"] for s in similar], top_k=10)

    category_recommendations = []
    if category_stats is not None and len(category_stats) > 0:
        category_recommendations = (
            category_stats.head(10)
            .rename(columns={"mean": "avg_view_count"})
            .to_dict(orient="records")
        )
        for c in category_recommendations:
            c["count"] = int(c["count"])
            c["avg_view_count"] = float(c["avg_view_count"])
            c["median"] = float(c["median"])

    return jsonify(
        {
            "input": {"title": title, "category": category, "author_id": author_id, "publish_time": publish_time},
            "predicted_view_count": predicted,
            "predicted_bucket_name": predicted_bucket,
            "keywords": keywords,
            "similar_videos": similar[:5],
            "top_categories": category_recommendations,
        }
    )

if __name__ == '__main__':
    try:
        # 在应用启动前加载数据
        load_data()
        # 启动Flask应用
        app.run(host="0.0.0.0", debug=True, port=5000)
    except FileNotFoundError as e:
        print(f"应用启动失败: {e}")
