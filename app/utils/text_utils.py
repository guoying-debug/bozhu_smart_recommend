import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def get_stopwords():
    return {
        "的", "是", "了", "在", "我", "你", "他", "她", "它", "们", "都", "也",
        "一个", "怎么", "这种", "这个", "一下", "真的", "就是", "还是", "不是",
        "什么", "为什么", "可以", "没有", "以及", "但是", "因为", "如果", "所以",
        "【", "】", "！", "？", "，", "。", " ", "#", "视频", "内容", "今天",
        "大家", "非常", "那个", "那个", "然后", "接着", "最后", "开始"
    }

def jieba_tokenize(text: str):
    return jieba.lcut(text)

def preprocess_text_for_bert(text):
    """
    为BERT准备文本，这里可以做一些简单的清洗，但要保留大部分原始信息。
    """
    # 移除一些特殊符号，但保留中英文和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    return text.strip()

def top_keywords_from_titles(titles, top_k=10):
    """
    升级版关键词提取：使用 TF-IDF 加权，而不是简单的词频统计。
    """
    if not titles:
        return []
        
    stopwords = get_stopwords()
    
    # 预处理
    clean_titles = []
    for t in titles:
        t = "" if not t else str(t)
        t = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", " ", t).strip()
        words = [w for w in jieba.lcut(t) if w.strip() and w not in stopwords and len(w) > 1]
        clean_titles.append(" ".join(words))
    
    if not clean_titles:
        return []

    try:
        # 使用 TF-IDF 提取关键词
        # max_df=0.95: 忽略在超过 95% 的文档中出现的词（通用高频词）
        # min_df=2: 忽略只出现过一次的词
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=top_k*2)
        tfidf_matrix = vectorizer.fit_transform(clean_titles)
        feature_names = vectorizer.get_feature_names_out()
        
        # 计算所有文档的平均 TF-IDF 分数
        avg_scores = tfidf_matrix.mean(axis=0).A1
        
        # 获取分数最高的索引
        top_indices = avg_scores.argsort()[::-1][:top_k]
        
        return [feature_names[i] for i in top_indices]
    except ValueError:
        # 如果文档太少或者词汇表为空，回退到简单词频
        counts = {}
        for t in clean_titles:
            for w in t.split():
                counts[w] = counts.get(w, 0) + 1
        return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]
