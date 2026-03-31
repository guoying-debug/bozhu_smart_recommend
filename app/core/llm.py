import os
import json
import dashscope
from dashscope import Generation
import logging

# 【重构】prompt 模板已迁移至 prompts/ 目录，通过 loader 加载，与业务逻辑解耦。
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def analyze_title_with_llm(title: str, category: str, predicted_view: float,
                           feature_explanations: list, similar_titles: list, feature_importance: list = None):
    """
    使用通义千问 (qwen-plus) 对标题进行深度分析和优化建议。
    将传统模型的预测结果、特征解释、相似爆款作为 Context 提供给 LLM。
    """
    if not dashscope.api_key:
        logger.warning("未检测到 DASHSCOPE_API_KEY，跳过 LLM 分析")
        return {
            "analysis": "未配置 LLM 服务，无法生成深度分析。",
            "suggestions": ["请检查环境变量配置"]
        }

    similar_titles_str = "\n".join(
        [f"- {t['title']} ({int(t['view_count']/10000)}万播放)" for t in similar_titles[:3]]
    )
    features_str = "\n".join(
        [f"- {exp['feature']}: {exp['effect']} ({exp['reason']})" for exp in feature_explanations]
    )
    global_importance_str = ""
    if feature_importance:
        global_importance_str = "\n".join(
            [f"- {item['feature']}: {item['score']:.4f}" for item in feature_importance[:3]]
        )

    # 【重构】从 prompts/llm_analyze_title.txt 加载模板，不再硬编码在此
    template = load_prompt("llm_analyze_title")
    prompt = template.format(
        title=title,
        category=category,
        predicted_view=int(predicted_view),
        feature_explanations=features_str,
        global_importance=global_importance_str,
        similar_titles=similar_titles_str,
    )

    try:
        response = Generation.call(
            model=Generation.Models.qwen_plus,
            prompt=prompt,
            result_format='message'
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                return {"diagnosis": content, "suggestions": []}
        else:
            logger.error(f"LLM API 调用失败: {response.status_code} - {response.message}")
            return {"diagnosis": "LLM 服务异常", "suggestions": []}
    except Exception as e:
        logger.error(f"analyze_title_with_llm failed: {e}")
        return {"diagnosis": f"服务异常: {e}", "suggestions": []}


def generate_search_queries(title: str) -> list:
    """
    Query Rewriting: 生成多个语义相似的搜索查询，提高 RAG 召回率。
    """
    if not dashscope.api_key:
        return [title]

    # 【重构】从 prompts/query_rewrite.txt 加载模板
    template = load_prompt("query_rewrite")
    prompt = template.format(title=title)

    try:
        response = Generation.call(
            model=Generation.Models.qwen_plus,
            prompt=prompt,
            result_format='message'
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            try:
                queries = json.loads(content)
                if isinstance(queries, list):
                    return queries
            except Exception:
                pass
        return [title]
    except Exception as e:
        logger.error(f"Query Rewrite failed: {e}")
        return [title]


def generate_hyde_doc(title: str) -> str:
    """
    HyDE (Hypothetical Document Embeddings): 生成假设性的爆款标题/描述用于检索。
    """
    if not dashscope.api_key:
        return title

    # 【重构】从 prompts/hyde_doc.txt 加载模板
    template = load_prompt("hyde_doc")
    prompt = template.format(title=title)

    try:
        response = Generation.call(
            model=Generation.Models.qwen_plus,
            prompt=prompt,
            result_format='message'
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content.strip()
        return title
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return title
