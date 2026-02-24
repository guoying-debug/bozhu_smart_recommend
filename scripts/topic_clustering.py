import pandas as pd
from sqlalchemy import create_engine
import os
import jieba
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import re
import joblib
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# --- 数据库配置 ---
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "bilibili_data")
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- 模型和文件路径 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 文本预处理 ---
def get_stopwords():
    """返回一个包含常见中文停用词的集合。"""
    return {"的", "是", "了", "在", "我", "你", "他", "她", "它", "们", "都", "也",
            "【", "】", "！", "？", "，", "。", " ", "#", "这个", "一个", "怎么", "这种"}

def preprocess_text_for_bert(text):
    """
    为BERT准备文本，这里可以做一些简单的清洗，但要保留大部分原始信息。
    """
    # 移除一些特殊符号，但保留中英文和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    return text.strip()

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

# --- 可视化 ---
def plot_clusters(df, x_col, y_col, cluster_col, title, filename):
    """
    根据降维结果和聚类标签绘制散点图并保存。
    """
    print(f"正在生成可视化图表: {title}...")
    
    # 设置中文字体，防止乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(16, 10))
    
    noise = df[df[cluster_col] == -1]
    non_noise = df[df[cluster_col] != -1]
    
    sns.scatterplot(data=non_noise, x=x_col, y=y_col, hue=cluster_col,
                    palette=sns.color_palette("hsv", len(non_noise[cluster_col].unique())),
                    legend="full", alpha=0.8)
    
    if not noise.empty:
        sns.scatterplot(data=noise, x=x_col, y=y_col, color="grey", marker="x", label="Noise", alpha=0.5)

    plt.title(title, fontsize=16)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=cluster_col)
    
    filepath = os.path.join(MODEL_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"图表已保存到: {filepath}")

# --- 特征重要性分析 ---
def analyze_feature_importance(df):
    """
    使用随机森林分析影响话题聚类的特征重要性。
    """
    print("\n--- 开始进行特征重要性分析 ---")
    
    # 选择特征和目标
    # 我们选择K-Means的聚类结果作为目标，因为它没有噪声点，更适合分类任务
    features = ['view_count', 'like_count', 'coin_count', 'favorite_count', 'share_count']
    target = 'bert_kmeans_cluster'
    
    X = df[features]
    y = df[target]
    
    print(f"分析特征: {features}")
    print(f"目标变量: {target}")
    
    # 训练随机森林模型
    print("正在训练随机森林分类器...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print("模型训练完成。")
    
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
    
    filepath = os.path.join(MODEL_DIR, 'feature_importance.png')
    plt.savefig(filepath)
    plt.close()
    print(f"\n特征重要性图表已保存到: {filepath}")
    print("-" * 50)


def perform_clustering():
    """
    加载数据，执行文本预处理、BERT向量化、聚类、降维、可视化和特征分析，并保存所有结果。
    """
    if not DB_PASSWORD:
        print("致命错误：未设置 DB_PASSWORD 环境变量。")
        return

    try:
        # 1. 加载数据
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT video_id, title, view_count, like_count, coin_count, favorite_count, share_count FROM videos", engine)
        df = df.reset_index(drop=True)
        print(f"成功从数据库加载了 {len(df)} 条视频数据。")
        print("-" * 50)

        # 2. 文本预处理
        print("正在为BERT进行文本预处理...")
        df['clean_title'] = df['title'].apply(preprocess_text_for_bert)
        print("文本预处理完成。")
        print("-" * 50)

        # 3. BERT 向量化
        embeddings_path = os.path.join(MODEL_DIR, 'bert_embeddings.npy')
        text_vectors = None
        if os.path.exists(embeddings_path):
            print("正在加载已保存的BERT向量...")
            loaded_vectors = np.load(embeddings_path)
            if loaded_vectors.shape[0] == len(df):
                text_vectors = loaded_vectors
            else:
                print(f"已保存向量数量({loaded_vectors.shape[0]})与数据条数({len(df)})不一致，将重新生成BERT向量...")

        if text_vectors is None:
            print("开始生成新的BERT向量...")
            text_vectors = get_bert_embeddings(df['clean_title'].tolist())
            np.save(embeddings_path, text_vectors)
            print(f"BERT向量已生成并保存到 {embeddings_path}")
        print("BERT向量化完成。向量矩阵形状:", text_vectors.shape)
        print("-" * 50)
        
        # 4. K-Means 聚类
        num_clusters = 10
        print(f"正在进行K-Means聚类（{num_clusters}个聚类）...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df['bert_kmeans_cluster'] = kmeans.fit_predict(text_vectors)
        print("K-Means聚类完成。")
        joblib.dump(kmeans, os.path.join(MODEL_DIR, 'bert_kmeans_model.pkl'))
        print("-" * 50)

        # 5. DBSCAN 聚类
        print("正在进行DBSCAN聚类...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        text_vectors_scaled = scaler.fit_transform(text_vectors)
        dbscan = DBSCAN(eps=20, min_samples=5, metric='euclidean')
        df['dbscan_cluster'] = dbscan.fit_predict(text_vectors_scaled)
        print("DBSCAN聚类完成。")
        joblib.dump(dbscan, os.path.join(MODEL_DIR, 'bert_dbscan_model.pkl'))
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

        # 7. 可视化
        plot_clusters(df, 'tsne_x', 'tsne_y', 'bert_kmeans_cluster', 'K-Means Clustering Visualization (t-SNE)', 'kmeans_tsne_visualization.png')
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
        output_path = os.path.join(MODEL_DIR, 'videos_with_clusters_and_coords.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"包含聚类和降维坐标的最终数据已保存到 {output_path}")
        
        # 10. 特征重要性分析
        analyze_feature_importance(df)

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    perform_clustering()
