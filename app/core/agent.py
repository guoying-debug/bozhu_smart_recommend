import os
import logging

# 【修复】删除了原有的 sys.path.append hack。
# 根因：agent.py 作为 app 包的一部分，只要通过 app.py 或 web_agent.py 正确启动，
# Python 的包查找机制就能找到 app.*，不需要手动篡改 sys.path。

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_openai_tools_agent, AgentExecutor

from app.core.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from app.skills import build_skills
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)

# 【修复】原代码在模块顶层直接实例化 llm / memory / rag_db / tools / agent_executor，
# 导致 import 时就发起 API 网络连接，在无 key 或无网络环境下无法导入该模块（单测无法运行）。
# 现在改为懒加载：所有对象在第一次调用 run_bilibili_agent() 时才初始化。
_agent_executor = None


def _make_tools(llm):
    """从 skills 层构造工具列表，新增技能只需修改 app/skills/__init__.py。"""
    return [skill.as_tool() for skill in build_skills(llm)]


def init_bilibili_agent():
    """
    构建并返回 AgentExecutor。
    【修复】原代码在模块顶层直接构建，现改为函数，在首次调用时才实例化。
    """
    if not LLM_API_KEY:
        raise RuntimeError("未配置 ZHIPU_API_KEY/LLM_API_KEY，无法初始化 Agent")

    llm = ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
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
