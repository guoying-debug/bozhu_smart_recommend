import os
import json
import dashscope
from dashscope import Generation
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 从环境变量获取 API KEY (无需硬编码，假设环境已配置 DASHSCOPE_API_KEY)
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

    # 1. 构建 Prompt Context
    similar_titles_str = "\n".join([f"- {t['title']} ({int(t['view_count']/10000)}万播放)" for t in similar_titles[:3]])
    
    features_str = "\n".join([f"- {exp['feature']}: {exp['effect']} ({exp['reason']})" for exp in feature_explanations])
    
    global_importance_str = ""
    if feature_importance:
        global_importance_str = "\n".join([f"- {item['feature']}: {item['score']:.4f}" for item in feature_importance[:3]])
    
    prompt = f"""
你是一个 B站爆款视频标题专家。请根据以下数据，为用户提供深度的标题诊断和优化建议。

【当前输入】
- 标题：{title}
- 分区：{category}
- 预测模型预估播放量：{int(predicted_view):,}
- 预测模型特征分析：
{features_str}

【全局特征重要性（影响话题聚类的关键因素）】
{global_importance_str}

【数据库中的相似爆款（作为参考标准）】
{similar_titles_str}

【任务要求】
1. **深度诊断**：结合预测模型的特征分析，指出该标题的优缺点。不要只说“短”，要分析情绪、受众、点击欲望。
2. **优化建议**：给出 3 个具体的优化方向，每个方向给出一个修改后的示例标题。
3. **风格匹配**：确保建议符合 B站 {category} 区的调性（例如知识区要硬核/悬念，生活区要真实/共鸣）。
4. **参考全局特征**：如果提供了全局特征重要性，请在分析中提及哪些指标（如互动率、分享数）对于成为热门话题至关重要，并建议如何在标题或内容中引导这些互动。

请以 JSON 格式返回结果，包含两个字段：
- `diagnosis`: 字符串，诊断报告。
- `suggestions`: 字符串列表，3个优化后的标题建议。
"""

    try:
        # 2. 调用通义千问 API
        response = Generation.call(
            model=Generation.Models.qwen_plus,
            prompt=prompt,
            result_format='message'  # 返回格式
        )

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # 尝试解析 JSON，如果 LLM 返回了 Markdown 代码块，需要清洗
            content = content.replace("```json", "").replace("```", "").strip()
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # 如果 LLM 没按 JSON 返回，直接作为文本处理
                logger.warning("LLM 返回格式非 JSON，进行降级处理")
                return {
                    "diagnosis": content,
                    "suggestions": []
                }
        else:
            logger.error(f"LLM 调用失败: {response.code} - {response.message}")
            return {
                "diagnosis": "智能体服务暂时不可用。",
                "suggestions": []
            }

    except Exception as e:
        logger.error(f"LLM 处理异常: {e}")
        return {
            "diagnosis": "分析过程发生错误。",
            "suggestions": []
        }
