from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import jieba
from app.core.data_loader import get_data
from app.utils.text_utils import jieba_tokenize, get_stopwords

_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None

def init_recommender():
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX
    df = get_data()
    if df is None:
        return

    print("正在构建 TF-IDF 矩阵...")
    stopwords = get_stopwords()
    
    # 简单的分词 wrapper
    def tokenizer(text):
        return [w for w in jieba.lcut(text) if w not in stopwords and len(w) > 1]

    # Create vectorizer
    _TFIDF_VECTORIZER = TfidfVectorizer(tokenizer=tokenizer, max_features=5000)
    
    # Fill NaN
    titles = df['title'].fillna("").astype(str).tolist()
    _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(titles)
    print("TF-IDF 矩阵构建完成。")

def get_similar_titles(title: str, top_k=5):
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX
    if _TFIDF_VECTORIZER is None or _TFIDF_MATRIX is None:
        init_recommender()
        
    if _TFIDF_VECTORIZER is None:
        return []
        
    df = get_data()
    
    try:
        query_vec = _TFIDF_VECTORIZER.transform([title])
        sims = cosine_similarity(query_vec, _TFIDF_MATRIX).flatten()
        
        # Get top indices
        top_indices = sims.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            # 降低阈值到 0.05，或者如果结果为空，允许更低
            if score < 0.05: 
                continue
            
            row = df.iloc[idx]
            results.append({
                "title": row['title'],
                "view_count": int(row['view_count']),
                "similarity": score,
                "cluster": int(row['bert_kmeans_cluster']) if 'bert_kmeans_cluster' in row else -1
            })
            
        # 兜底逻辑：如果找不到相似的，返回同类别或全局Top热度视频
        if not results and not df.empty:
             # 简单兜底：随机返回 3 个高播放量视频作为“热门参考”
             top_videos = df.nlargest(20, 'view_count').sample(min(3, len(df)))
             for _, row in top_videos.iterrows():
                 results.append({
                    "title": f"[热门兜底] {row['title']}",
                    "view_count": int(row['view_count']),
                    "similarity": 0.0,
                    "cluster": int(row['bert_kmeans_cluster']) if 'bert_kmeans_cluster' in row else -1
                 })
            
        return results
    except Exception as e:
        print(f"推荐计算出错: {e}")
        return []
