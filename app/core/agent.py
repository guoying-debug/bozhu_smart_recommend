import os
from typing import List
import json
import logging

# 【修复】删除了原有的 sys.path.append hack。
# 根因：agent.py 作为 app 包的一部分，只要通过 app.py 或 web_agent.py 正确启动，
# Python 的包查找机制就能找到 app.*，不需要手动篡改 sys.path。

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from langchain.schema import SystemMessage
from langchain_core.runnables import RunnablePassthrough

from app.core.recommender import get_similar_titles
from app.core.predictor import predict_view
from app.models.schemas import PredictRequest
# 【重构】agent system prompt 已迁移至 prompts/agent_system.txt，通过 loader 加载
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)

# 【修复】原代码在模块顶层直接实例化 llm / memory / rag_db / tools / agent_executor，
# 导致 import 时就发起 API 网络连接，在无 key 或无网络环境下无法导入该模块（单测无法运行）。
# 现在改为懒加载：所有对象在第一次调用 run_bilibili_agent() 时才初始化。
_agent_executor = None


def _make_tools(llm):
    """构造 Agent 使用的工具列表，依赖注入 llm 而非全局变量。"""

    def chat_tool(query: str) -> str:
        """闲聊工具：处理问候、感谢等无业务诉求的对话"""
        greet_words = ["你好", "哈喽", "嗨", "早上好", "下午好", "晚上好"]
        thank_words = ["谢谢", "感谢", "多谢", "辛苦了"]
        if any(word in query for word in greet_words):
            return "你好呀！我是B站运营智能助手，能帮你优化标题、预测播放量，有任何问题都可以问～"
        elif any(word in query for word in thank_words):
            return "不客气啦！能帮到你我超开心，有其他问题随时都可以问～"
        else:
            return "哈哈，你是不是想和我闲聊呀？尽管说～"

    def title_optimize_tool(query: str) -> str:
        """标题优化工具：基于现有爆款视频库进行RAG检索，并给出优化建议。"""
        similar_titles = get_similar_titles(query, top_k=3)
        similar_titles_str = "\n".join(
            [f"- {t['title']} ({int(t['view_count']/10000)}万播放)" for t in similar_titles]
        )
        # 【重构】从 prompts/title_optimize.txt 加载模板
        template = load_prompt("title_optimize")
        prompt = PromptTemplate(
            template=template,
            input_variables=["docs", "query"]
        )
        chain = prompt | llm
        response = chain.invoke({"docs": similar_titles_str, "query": query})
        return response.content

    def play_volume_predict_tool(title: str) -> str:
        """播放量预测工具：调用现有机器学习模型预测播放量"""
        try:
            req = PredictRequest(title=title, category="未知")
            pred_view, bucket_id, bucket_name_str, explanations = predict_view(req)
            features_str = "\n".join(
                [f"- {exp['feature']}: {exp['effect']} ({exp['reason']})" for exp in explanations]
            )
            return f"""模型预测结果如下：
- 预估播放量：{int(pred_view):,} 次
- 所属档位：{bucket_name_str}

特征分析：
{features_str}"""
        except Exception as e:
            return f"预测失败: {str(e)}"

    return [
        Tool(name="闲聊工具", func=chat_tool,
             description="仅用于处理用户的问候、感谢等闲聊类请求，不涉及任何业务分析"),
        Tool(name="标题优化工具", func=title_optimize_tool,
             description="当用户需要优化B站视频标题时调用。输入用户提供的原始标题（可包含特殊要求）"),
        Tool(name="播放量预测工具", func=play_volume_predict_tool,
             description="当用户需要预测某个标题的B站播放量时调用。输入视频标题"),
    ]


def init_bilibili_agent():
    """
    构建并返回 AgentExecutor。
    【修复】原代码在模块顶层直接构建，现改为函数，在首次调用时才实例化。
    """
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        temperature=0.7,
    )

    # 【修复】memory 不再是全局单例，每次构建 agent 时创建新实例，
    # 避免多用户/多请求共享同一段对话历史导致上下文污染。
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
        output_key="output"
    )

    tools = _make_tools(llm)

    # 【重构】从 prompts/agent_system.txt 加载 system prompt，不再硬编码
    system_prompt = load_prompt("agent_system")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors="无法理解你的问题，请重新描述",
        return_intermediate_steps=True
    )
    return agent_executor


def run_bilibili_agent(query: str) -> str:
    """调用B站运营Agent处理用户查询。首次调用时懒加载初始化。"""
    global _agent_executor
    if _agent_executor is None:
        logger.info("首次调用，正在初始化 Bilibili Agent...")
        _agent_executor = init_bilibili_agent()
    try:
        response = _agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        logger.error(f"Agent 处理失败: {e}")
        return f"Agent处理失败：{str(e)}"


if __name__ == "__main__":
    print("=== B站运营Agent启动（输入q退出）===")
    while True:
        user_query = input("请输入你的问题：")
        if user_query.lower() == "q":
            print("Agent已退出～")
            break
        result = run_bilibili_agent(user_query)
        print(f"\nAgent回复：{result}\n" + "-"*50)
