import os

import pandas as pd
import numpy as np
import chromadb
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from app.core.data_loader import load_data
from app.core.config import CHROMA_DB_DIR
from app.core.llm import generate_search_queries, generate_hyde_doc
from app.utils.text_utils import get_stopwords

_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None
_BERT_TOKENIZER = None
_BERT_MODEL = None
_DEVICE = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None
_TITLES_LIST = None

def init_recommender():
    """
    初始化 ChromaDB (Dense), BERT (Dense), 和 TF-IDF (Sparse)。
    """
    global _CHROMA_CLIENT, _CHROMA_COLLECTION, _BERT_TOKENIZER, _BERT_MODEL, _DEVICE
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _TITLES_LIST

    # 设置 Hugging Face 镜像：放在函数内而非模块顶层，
    # 避免 import 时污染全局环境（影响其他不需要此镜像的模块）
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    # 1. 初始化 ChromaDB
    if _CHROMA_CLIENT is None:
        print(f"正在连接 ChromaDB (路径: {CHROMA_DB_DIR})...")
        try:
            if not os.path.exists(CHROMA_DB_DIR):
                print("ChromaDB 目录不存在，请先运行离线脚本 topic_clustering.py 进行数据入库。")
            else:
                _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                try:
                    _CHROMA_COLLECTION = _CHROMA_CLIENT.get_collection(name="bilibili_videos")
                    print(f"成功加载 ChromaDB 集合: {_CHROMA_COLLECTION.count()} 条数据")
                except Exception:
                    print("未找到 bilibili_videos 集合，请检查离线入库流程。")
        except Exception as e:
            print(f"ChromaDB 初始化失败: {e}")

    # 2. 初始化 BERT 模型
    if _BERT_MODEL is None:
        print("正在加载 BERT 模型用于在线推理...")
        try:
            # 增加本地路径或指定镜像参数
            model_name = 'bert-base-chinese'
            _BERT_TOKENIZER = BertTokenizer.from_pretrained(model_name, local_files_only=False)
            _BERT_MODEL = BertModel.from_pretrained(model_name, local_files_only=False)
            
            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _BERT_MODEL.to(_DEVICE)
            _BERT_MODEL.eval()
            print(f"BERT 模型加载完成 (Device: {_DEVICE})")
        except Exception as e:
            print(f"BERT 模型加载失败: {e}")
            # 尝试再次强制设置环境变量后重试
            try:
                print("尝试备用镜像源加载...")
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                _BERT_TOKENIZER = BertTokenizer.from_pretrained(model_name, mirror='tuna')
                _BERT_MODEL = BertModel.from_pretrained(model_name, mirror='tuna')
                _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _BERT_MODEL.to(_DEVICE)
                _BERT_MODEL.eval()
                print(f"BERT 模型备用加载完成 (Device: {_DEVICE})")
            except Exception as e2:
                print(f"BERT 模型备用加载也失败: {e2}")

    # 3. 初始化 TF-IDF (Sparse Retrieval)
    if _TFIDF_VECTORIZER is None:
        print("正在构建 TF-IDF 索引 (Sparse Retrieval)...")
        df = load_data()
        if df is not None and not df.empty:
            stopwords = get_stopwords()
            def tokenizer(text):
                return [w for w in jieba.lcut(text) if w not in stopwords and len(w) > 1]
            
            try:
                _TFIDF_VECTORIZER = TfidfVectorizer(tokenizer=tokenizer, max_features=10000)
                _TITLES_LIST = df['title'].fillna("").astype(str).tolist()
                _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(_TITLES_LIST)
                print("TF-IDF 索引构建完成。")
            except Exception as e:
                print(f"TF-IDF 构建失败: {e}")

def get_bert_embedding_single(text: str):
    """
    为单个标题生成 BERT 向量
    """
    global _BERT_TOKENIZER, _BERT_MODEL, _DEVICE
    
    if _BERT_MODEL is None:
        init_recommender()
        
    if _BERT_MODEL is None:
        raise RuntimeError("BERT 模型未初始化")

    inputs = _BERT_TOKENIZER(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(_DEVICE) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = _BERT_MODEL(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding.tolist()

def _dense_search(queries, top_k=10):
    """
    使用 ChromaDB 进行稠密向量检索。支持多个查询向量。
    """
    if _CHROMA_COLLECTION is None:
        return {}
    
    results_map = {} # title -> score (distance)
    
    for query in queries:
        try:
            vec = get_bert_embedding_single(query)
            res = _CHROMA_COLLECTION.query(
                query_embeddings=[vec],
                n_results=top_k,
                include=['metadatas', 'distances']
            )
            
            metadatas = res['metadatas'][0]
            distances = res['distances'][0]
            
            for i, meta in enumerate(metadatas):
                title = meta['title']
                dist = distances[i]
                # 简单归一化: score = 1 / (1 + dist)
                score = 1.0 / (1.0 + dist)
                
                # 保留最高分
                if title not in results_map or score > results_map[title]['score']:
                    results_map[title] = {
                        'score': score,
                        'meta': meta,
                        'source': 'dense'
                    }
        except Exception as e:
            print(f"Dense search error for '{query}': {e}")
            
    return results_map

def _sparse_search(queries, top_k=10):
    """
    使用 TF-IDF 进行稀疏检索。
    """
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX, _TITLES_LIST
    
    if _TFIDF_VECTORIZER is None:
        return {}
        
    results_map = {}
    df = load_data()
    
    for query in queries:
        try:
            query_vec = _TFIDF_VECTORIZER.transform([query])
            sims = cosine_similarity(query_vec, _TFIDF_MATRIX).flatten()
            
            # 获取 Top K
            top_indices = sims.argsort()[::-1][:top_k]
            
            for idx in top_indices:
                score = float(sims[idx])
                if score < 0.05: continue
                
                title = _TITLES_LIST[idx]
                row = df.iloc[idx]
                
                if title not in results_map or score > results_map[title]['score']:
                    results_map[title] = {
                        'score': score,
                        'meta': {
                            "title": row['title'],
                            "view_count": int(row['view_count']),
                            "category": row['category'] if 'category' in row else "未知",
                            "cluster_id": int(row['bert_kmeans_cluster']) if 'bert_kmeans_cluster' in row else -1
                        },
                        'source': 'sparse'
                    }
        except Exception as e:
            print(f"Sparse search error: {e}")
            
    return results_map

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    RRF 融合算法: score = 1 / (k + rank)
    """
    fused_scores = {}
    
    # 处理 Dense 结果
    # dense_results 是 dict: title -> {score, meta}
    # 我们先按 score 排序得到 rank
    sorted_dense = sorted(dense_results.items(), key=lambda x: x[1]['score'], reverse=True)
    for rank, (title, data) in enumerate(sorted_dense):
        if title not in fused_scores:
            fused_scores[title] = {'rrf_score': 0.0, 'meta': data['meta'], 'sources': set()}
        fused_scores[title]['rrf_score'] += 1.0 / (k + rank + 1)
        fused_scores[title]['sources'].add('dense')
        
    # 处理 Sparse 结果
    sorted_sparse = sorted(sparse_results.items(), key=lambda x: x[1]['score'], reverse=True)
    for rank, (title, data) in enumerate(sorted_sparse):
        if title not in fused_scores:
            fused_scores[title] = {'rrf_score': 0.0, 'meta': data['meta'], 'sources': set()}
        fused_scores[title]['rrf_score'] += 1.0 / (k + rank + 1)
        fused_scores[title]['sources'].add('sparse')
        
    # 转换为列表并排序
    final_results = []
    for title, data in fused_scores.items():
        final_results.append({
            "title": title,
            "view_count": data['meta']['view_count'],
            "similarity": data['rrf_score'], # RRF score 只是排序依据，不是严格的相似度
            "cluster": data['meta']['cluster_id'],
            "category": data['meta']['category'],
            "sources": list(data['sources'])
        })
        
    final_results.sort(key=lambda x: x['similarity'], reverse=True)
    return final_results

def get_similar_titles(title: str, top_k=5):
    """
    混合检索 (Hybrid Retrieval) + RAG 增强
    1. Query Rewriting & HyDE
    2. Dense Search (BERT+Chroma)
    3. Sparse Search (TF-IDF)
    4. RRF Fusion
    """
    init_recommender()
    
    # 1. 扩展查询 (Query Expansion)
    search_queries = [title]
    
    # 使用 LLM 生成改写查询 (Query Rewriting)
    rewritten_queries = generate_search_queries(title)
    if rewritten_queries:
        print(f"LLM 改写查询: {rewritten_queries}")
        search_queries.extend(rewritten_queries)
        
    # 使用 LLM 生成假设性文档 (HyDE)
    hyde_doc = generate_hyde_doc(title)
    if hyde_doc and hyde_doc != title:
        print(f"HyDE 文档: {hyde_doc[:30]}...")
        search_queries.append(hyde_doc)
        
    # 去重
    search_queries = list(set(search_queries))
    
    # 2. 执行检索
    # Dense: 语义匹配
    dense_hits = _dense_search(search_queries, top_k=top_k*2)
    
    # Sparse: 关键词匹配
    sparse_hits = _sparse_search(search_queries, top_k=top_k*2)
    
    # 3. 结果融合 (RRF)
    if not dense_hits and not sparse_hits:
        return get_fallback_recommendations(top_k)
        
    fused_results = reciprocal_rank_fusion(dense_hits, sparse_hits)
    
    # 返回 Top K
    return fused_results[:top_k]

def get_fallback_recommendations(top_k=5):
    """
    兜底策略：如果向量检索失败，返回热门视频。
    """
    print("触发兜底推荐策略...")
    df = load_data()
    if df is None or df.empty:
        return []
        
    top_videos = df.nlargest(20, 'view_count').sample(min(top_k, len(df)))
    results = []
    for _, row in top_videos.iterrows():
        results.append({
        "title": f"[热门兜底] {row['title']}",
        "view_count": int(row['view_count']),
        "similarity": 0.0,
        "cluster": int(row['bert_kmeans_cluster']) if 'bert_kmeans_cluster' in row else -1,
        "category": row['category'] if 'category' in row else "未知",
        "sources": ["fallback"]
        })
    return results
