import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from app.core.data_loader import get_data
from app.utils.text_utils import top_keywords_from_titles

_FEATURE_IMPORTANCE_CACHE = None

def get_feature_importance():
    global _FEATURE_IMPORTANCE_CACHE
    if _FEATURE_IMPORTANCE_CACHE is not None:
        return _FEATURE_IMPORTANCE_CACHE
        
    df = get_data()
    if df is None:
        return []
        
    features = ['view_count', 'like_count', 'coin_count', 'favorite_count', 'share_count']
    target = 'bert_kmeans_cluster'
    
    if target not in df.columns or df[target].isnull().all():
        return []
        
    # 移除目标列为NaN的行
    df_cleaned = df.dropna(subset=[target])
    
    # 确保特征列存在
    valid_features = [f for f in features if f in df_cleaned.columns]
    if not valid_features:
        return []

    X = df_cleaned[valid_features]
    y = df_cleaned[target]
    
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importance = []
        for name, score in zip(valid_features, rf.feature_importances_):
            importance.append({"feature": name, "score": float(score)})
            
        importance.sort(key=lambda x: x['score'], reverse=True)
        _FEATURE_IMPORTANCE_CACHE = importance
        return importance
    except Exception as e:
        print(f"特征重要性分析失败: {e}")
        return []

def get_cluster_summary():
    df = get_data()
    if df is None:
        return {}
        
    summary = {}
    
    # K-Means 摘要
    if 'bert_kmeans_cluster' in df.columns:
        kmeans_counts = df['bert_kmeans_cluster'].value_counts().to_dict()
        summary['kmeans'] = {int(k): int(v) for k, v in kmeans_counts.items()}
        
    # DBSCAN 摘要
    if 'dbscan_cluster' in df.columns:
        n_noise = len(df[df['dbscan_cluster'] == -1])
        n_clusters = len(df['dbscan_cluster'].unique()) - (1 if n_noise > 0 else 0)
        summary['dbscan'] = {
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'noise_ratio': float(n_noise / len(df))
        }
        
    return summary

def get_topics_list():
    df = get_data()
    if df is None:
        return []
        
    if 'bert_kmeans_cluster' not in df.columns:
        return []

    topics = []
    # 按播放量降序排列聚类
    cluster_stats = df.groupby('bert_kmeans_cluster')['view_count'].sum().sort_values(ascending=False)
    
    for cluster_id in cluster_stats.index:
        sub_df = df[df['bert_kmeans_cluster'] == cluster_id]
        
        # 提取关键词
        titles = sub_df['title'].tolist()
        keywords = top_keywords_from_titles(titles, top_k=8)
        
        # 提取代表性视频（播放量Top5）
        top_videos_df = sub_df.sort_values('view_count', ascending=False).head(5)
        top_videos = []
        for _, row in top_videos_df.iterrows():
            top_videos.append({
                "title": row['title'],
                "view_count": int(row['view_count'])
            })
        
        topics.append({
            "cluster_id": int(cluster_id),
            "total_videos": int(len(sub_df)),
            "total_views": int(sub_df['view_count'].sum()),
            "average_views": float(sub_df['view_count'].mean()),
            "top_terms": keywords,
            "top_videos": top_videos
        })
        
    return topics
