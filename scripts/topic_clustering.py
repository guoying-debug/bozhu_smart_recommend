import pandas as pd
from sqlalchemy import create_engine
import os
# 设置 Hugging Face 镜像，解决连接超时问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import jieba
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import re
import joblib
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import chromadb

# 添加项目根目录到 sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.core.config import (
    get_database_url, DATA_PROCESSED_DIR, MODELS_TRAINED_DIR, IMAGES_DIR,
    BERT_EMBEDDINGS_PATH, BERT_KMEANS_MODEL_PATH, BERT_DBSCAN_MODEL_PATH,
    DATA_PATH, CHROMA_DB_DIR
)
from app.utils.text_utils import preprocess_text_for_bert

# 确保目录存在
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_TRAINED_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- BERT 向量化 ---
def get_bert_embeddings(texts, model_name='bert-base-chinese', batch_size=16):
    """
    使用指定的BERT模型为文本列表生成向量。
    """
    print(f"正在加载BERT模型: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"BERT模型已加载到: {device}")

    model.eval()
    embeddings = []
    
    print("正在生成BERT向量...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 我们使用[CLS] token的输出来代表整个句子的向量
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
        
        if (i // batch_size) % 10 == 0:
            print(f"  已处理 {i + len(batch_texts)} / {len(texts)} 个标题")

    return np.vstack(embeddings)

# --- 自动选择最优 K 值 ---
def find_optimal_k(embeddings, k_range=range(5, 20)):
    print(f"正在寻找最优 K 值 (范围: {k_range})...")
    best_k = 0
    best_score = -1
    scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append(score)
        print(f"K={k}, Silhouette Score={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"最优 K 值: {best_k} (Score: {best_score:.4f})")
    
    # 绘制 K 值评估图
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), scores, marker='o')
    plt.title('Silhouette Score for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join(IMAGES_DIR, 'optimal_k_score.png'))
    plt.close()
    
    return best_k

# --- 交互式可视化 ---
def plot_clusters_interactive(df, x_col, y_col, cluster_col, title, filename):
    print(f"正在生成交互式可视化图表: {title}...")
    
    # 过滤噪声点用于着色
    df['Cluster_Label'] = df[cluster_col].apply(lambda x: f"Cluster {x}" if x != -1 else "Noise")
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color='Cluster_Label',
        hover_data=['title', 'view_count', 'category'],
        title=title,
        template="plotly_white",
        opacity=0.8
    )
    
    fig.update_layout(
        legend_title_text='Topic Clusters',
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2"
    )
    
    output_path = os.path.join(IMAGES_DIR, filename)
    fig.write_html(output_path)
    print(f"交互式图表已保存到: {output_path}")

# --- 特征重要性分析 ---
def analyze_feature_importance(df):
    """
    使用随机森林分析影响话题聚类的特征重要性。
    """
    print("\n--- 开始进行特征重要性分析 ---")
    
    features = ['view_count', 'like_count', 'coin_count', 'favorite_count', 'share_count']
    target = 'bert_kmeans_cluster'
    
    # 过滤掉缺失值
    df_clean = df.dropna(subset=features + [target])
    
    X = df_clean[features]
    y = df_clean[target]
    
    # 训练随机森林模型
    print("正在训练随机森林分类器...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 获取并展示特征重要性
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性排名:")
    print(importance_df)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance for Topic Clustering', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    filepath = os.path.join(IMAGES_DIR, 'feature_importance.png')
    plt.savefig(filepath)
    plt.close()
    print(f"\n特征重要性图表已保存到: {filepath}")
    print("-" * 50)

# --- ChromaDB 向量存储 ---
def save_to_chroma(df, embeddings):
    """
    将视频数据和 BERT 向量存入 ChromaDB，用于在线检索。
    """
    print("\n--- 开始将数据存入 ChromaDB ---")
    print(f"ChromaDB 存储路径: {CHROMA_DB_DIR}")
    
    # 确保目录存在
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    # 初始化 Chroma 客户端
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    collection_name = "bilibili_videos"
    
    # 获取或创建集合 (如果存在先删除以确保全量更新)
    try:
        client.delete_collection(name=collection_name)
        print(f"已删除旧集合: {collection_name}")
    except Exception:
        pass # 集合不存在或其他错误，忽略

        
    collection = client.create_collection(name=collection_name)
    print(f"已创建新集合: {collection_name}")
    
    # 准备数据
    ids = df['video_id'].astype(str).tolist()
    documents = df['title'].fillna("").astype(str).tolist()
    
    # 准备 Metadata
    metadatas = []
    for _, row in df.iterrows():
        meta = {
            "title": str(row['title']),
            "view_count": int(row['view_count']) if pd.notnull(row['view_count']) else 0,
            "category": str(row['category']) if pd.notnull(row['category']) else "未分类",
            "cluster_id": int(row.get('bert_kmeans_cluster', -1)),
            "video_id": str(row['video_id'])
        }
        metadatas.append(meta)
    
    # 批量插入
    batch_size = 100
    total = len(ids)
    
    print(f"正在插入 {total} 条向量数据...")
    embeddings_list = embeddings.tolist()
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_ids = ids[i:end]
        batch_embeddings = embeddings_list[i:end]
        batch_metadatas = metadatas[i:end]
        batch_documents = documents[i:end]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )
        if (i // batch_size) % 5 == 0:
            print(f"  已处理 {end}/{total} 条")
            
    print("ChromaDB 数据入库完成。")
    print("-" * 50)


def perform_clustering():
    """
    加载数据，执行文本预处理、BERT向量化、聚类、降维、可视化和特征分析，并保存所有结果。
    """
    database_url = get_database_url()
    if not database_url:
        print("致命错误：未设置 DB_PASSWORD 环境变量。")
        return

    try:
        # 1. 加载数据
        engine = create_engine(database_url)
        print("正在从数据库加载数据...")
        df = pd.read_sql("SELECT video_id, title, view_count, like_count, coin_count, favorite_count, share_count, category FROM videos", engine)
        df = df.reset_index(drop=True)
        print(f"成功从数据库加载了 {len(df)} 条视频数据。")
        print("-" * 50)

        if len(df) < 20:
            print("数据量过少，无法进行聚类。")
            return

        # 2. 文本预处理
        print("正在为BERT进行文本预处理...")
        df['clean_title'] = df['title'].apply(preprocess_text_for_bert)
        print("文本预处理完成。")
        print("-" * 50)

        # 3. BERT 向量化
        text_vectors = None
        if os.path.exists(BERT_EMBEDDINGS_PATH):
            print("正在加载已保存的BERT向量...")
            try:
                loaded_vectors = np.load(BERT_EMBEDDINGS_PATH)
                if loaded_vectors.shape[0] == len(df):
                    text_vectors = loaded_vectors
                else:
                    print(f"已保存向量数量({loaded_vectors.shape[0]})与数据条数({len(df)})不一致，将重新生成BERT向量...")
            except Exception as e:
                print(f"加载向量失败: {e}，将重新生成...")

        if text_vectors is None:
            print("开始生成新的BERT向量...")
            text_vectors = get_bert_embeddings(df['clean_title'].tolist())
            np.save(BERT_EMBEDDINGS_PATH, text_vectors)
            print(f"BERT向量已生成并保存到 {BERT_EMBEDDINGS_PATH}")
        print("BERT向量化完成。向量矩阵形状:", text_vectors.shape)
        print("-" * 50)
        
        # 4. 自动选择最优 K 值并聚类
        optimal_k = find_optimal_k(text_vectors, k_range=range(5, 15)) # 缩小范围以节省时间
        print(f"正在进行K-Means聚类（使用最优K={optimal_k}）...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['bert_kmeans_cluster'] = kmeans.fit_predict(text_vectors)
        print("K-Means聚类完成。")
        joblib.dump(kmeans, BERT_KMEANS_MODEL_PATH)
        print("-" * 50)

        # 5. DBSCAN 聚类
        print("正在进行DBSCAN聚类...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        text_vectors_scaled = scaler.fit_transform(text_vectors)
        dbscan = DBSCAN(eps=20, min_samples=5, metric='euclidean')
        df['dbscan_cluster'] = dbscan.fit_predict(text_vectors_scaled)
        print("DBSCAN聚类完成。")
        joblib.dump(dbscan, BERT_DBSCAN_MODEL_PATH)
        print("-" * 50)

        # 6. 降维
        print("正在进行PCA降维...")
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(text_vectors)
        df['pca_x'] = pca_result[:, 0]
        df['pca_y'] = pca_result[:, 1]
        print("PCA降维完成。")
        
        print("正在进行t-SNE降维（这可能需要一些时间）...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
        tsne_result = tsne.fit_transform(text_vectors)
        df['tsne_x'] = tsne_result[:, 0]
        df['tsne_y'] = tsne_result[:, 1]
        print("t-SNE降维完成。")
        print("-" * 50)

        # 7. 交互式可视化 (替换原静态图)
        plot_clusters_interactive(df, 'tsne_x', 'tsne_y', 'bert_kmeans_cluster', 'K-Means Clustering Visualization (t-SNE)', 'kmeans_tsne_interactive.html')
        print("-" * 50)

        # 8. 显示聚类结果摘要
        print("\n--- K-Means (BERT) 聚类结果摘要 ---")
        print(df['bert_kmeans_cluster'].value_counts())
        
        n_clusters_dbscan = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        n_noise = list(dbscan.labels_).count(-1)
        print(f"\n--- DBSCAN (BERT) 聚类结果摘要 ---")
        print(f"发现聚类数量: {n_clusters_dbscan}")
        print(f"噪声点数量: {n_noise} ({n_noise / len(df):.2%})")
        print(df['dbscan_cluster'].value_counts())
        print("-" * 50)
        
        # 9. 保存最终结果
        df.to_csv(DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"包含聚类和降维坐标的最终数据已保存到 {DATA_PATH}")
        
        # 10. 特征重要性分析
        analyze_feature_importance(df)
        
        # 11. 向量入库 (ChromaDB)
        save_to_chroma(df, text_vectors)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"发生错误: {e}")

if __name__ == "__main__":
    perform_clustering()
