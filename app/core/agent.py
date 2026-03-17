import os
import sys
from typing import List
import json

# 解决 ModuleNotFoundError: No module named 'app'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

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

# 导入项目中现有的功能
from app.core.recommender import get_similar_titles
from app.core.predictor import predict_view
from app.models.schemas import PredictRequest

# ====================== 2. 初始化基础组件 ======================
# 2.1 通义千问LLM（规划/决策核心），使用 openai 兼容接口调用阿里云 DashScope
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
)

# 2.2 会话记忆（保留最近5轮，支持上下文理解）
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"  # 匹配Agent输出格式
)

# 2.3 RAG向量库（标题优化/播放量预测的知识库）
def init_rag_db():
    """初始化B站运营知识库（标题优化/播放量预测相关）"""
    # 知识库内容（2026最新B站运营规则，可替换为本地文档）
    bilibili_docs = [
        "2026B站标题优化公式：AI+垂直领域+动作词（如“AI+Python自动整理Excel｜3行代码搞定”）",
        "B站标题避坑：纯标题党、无实质信息会被AI算法降权",
        "2026B站播放量预测因子：标题含搜索词（+30%）、中长视频（5-15分钟）（+25%）、AI互动组件（+40%）",
        "低粉UP主标题播放量区间：基础款（1000-5000）、优质款（5000-2万）、爆款潜质（2万+）",
        "标题含“AI+实操”关键词，播放量平均提升3.8倍",
        "B站播放量预测规则：无AI元素的标题，播放量普遍低于1000；含AI+垂类的标题，最低5000+"
    ]
    # 初始化嵌入模型（DashScope）
    embeddings = DashScopeEmbeddings(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
    # 创建Chroma向量库（轻量、本地运行，避免 FAISS 依赖报错）
    db = Chroma.from_texts(bilibili_docs, embeddings)
    return db

rag_db = init_rag_db()

# ====================== 3. 定义3个核心工具 ======================
# 工具1：闲聊工具（匹配问候/感谢类话术）
def chat_tool(query: str) -> str:
    """闲聊工具：处理问候、感谢等无业务诉求的对话"""
    greet_words = ["你好", "哈喽", "嗨", "早上好", "下午好", "晚上好"]
    thank_words = ["谢谢", "感谢", "多谢", "辛苦了"]
    
    if any(word in query for word in greet_words):
        return "你好呀 😊！我是B站运营智能助手，能帮你优化标题、预测播放量，有任何问题都可以问～"
    elif any(word in query for word in thank_words):
        return "不客气啦 😜！能帮到你我超开心，有其他问题随时都可以问～"
    else:
        return "哈哈，你是不是想和我闲聊呀？尽管说～"

# 工具2：标题优化工具（结合现有的RAG检索和预测模型）
def title_optimize_tool(query: str) -> str:
    """标题优化工具：基于现有爆款视频库进行RAG检索，并给出优化建议。如果用户提供了要求（如：侧重纯小白），请一并处理。"""
    # 提取标题和上下文要求（简单处理，把整个 query 传给检索也是可以的，但更精确的是传给大模型）
    similar_titles = get_similar_titles(query, top_k=3)
    
    similar_titles_str = "\n".join([f"- {t['title']} ({int(t['view_count']/10000)}万播放)" for t in similar_titles])
    
    # 2. 调用LLM生成优化建议
    prompt = PromptTemplate(
        template="""你是一个B站爆款标题专家。
以下是知识库中找到的与用户输入最相似的真实爆款视频标题：
{docs}

请基于上述爆款规律，优化用户提供的B站标题，要求：
1. 给出3条优化建议，每条标注核心优化点；
2. 风格要吸引人，符合B站调性；
3. 参考爆款标题的格式（如疑问句、数字、悬念等）；
4. 【重要】如果用户在输入中提出了特殊要求（例如：侧重纯小白教学、要搞笑一点等），请务必在生成标题时体现这些要求！

用户输入（包含原标题和可能的特殊要求）：{query}
优化建议：""",
        input_variables=["docs", "query"]
    )
    chain = prompt | llm
    response = chain.invoke({"docs": similar_titles_str, "query": query})
    return response.content

# 工具3：播放量预测工具（调用现有的预测模型）
def play_volume_predict_tool(title: str) -> str:
    """播放量预测工具：调用现有机器学习模型预测播放量"""
    try:
        req = PredictRequest(title=title, category="未知")
        pred_view, bucket_id, bucket_name, explanations = predict_view(req)
        
        features_str = "\n".join([f"- {exp['feature']}: {exp['effect']} ({exp['reason']})" for exp in explanations])
        
        return f"""
        模型预测结果如下：
        - 预估播放量：{int(pred_view):,} 次
        - 所属档位：{bucket_name}
        
        特征分析：
        {features_str}
        """
    except Exception as e:
        return f"预测失败: {str(e)}"

# 封装工具列表（Agent会自动匹配调用）
tools = [
    Tool(
        name="闲聊工具",
        func=chat_tool,
        description="仅用于处理用户的问候、感谢等闲聊类话术，比如：你好、谢谢、哈喽等，禁止用于业务问题"
    ),
    Tool(
        name="标题优化建议",
        func=title_optimize_tool,
        description="仅用于用户提供B站标题后，给出优化建议，必须调用RAG知识库，禁止直接回答"
    ),
    Tool(
        name="播放量预测",
        func=play_volume_predict_tool,
        description="仅用于用户提供B站标题后，预测播放量区间并说明理由，必须调用RAG知识库，禁止直接回答"
    )
]

# ====================== 4. 初始化Agent（规划+工具调用+记忆） ======================
def init_bilibili_agent():
    """初始化B站运营Agent（强制工具调用，避免LLM自作聪明）"""
    # 加载Agent基础Prompt（适配通义千问）
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # 自定义系统消息（强制工具调用规则，避免幻觉）
    system_message = SystemMessage(
        content="""你是严格遵循流程的B站运营Agent，必须执行以下规则：
1. 所有问题必须先匹配工具，禁止直接用内置知识回答：
   - 闲聊类（问候/感谢）→ 必须调用「闲聊工具」；
   - 标题优化→ 必须调用「标题优化建议」工具；
   - 播放量预测→ 必须调用「播放量预测」工具；
2. 即使你认为自己知道答案，也必须调用对应工具，回答完全基于工具返回结果；
3. 无匹配工具时，仅告知“暂无对应工具处理该问题”，禁止编造答案；"""
    )
    prompt.messages[0] = system_message  # 替换默认系统消息
    
    # 创建Agent（绑定LLM+工具+Prompt）
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # 创建Agent执行器（绑定记忆，开启调试日志）
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,  # 打印Agent思考/调用流程（调试用）
        handle_parsing_errors="无法理解你的问题，请重新描述",
        return_intermediate_steps=True  # 返回中间步骤（便于调试）
    )
    return agent_executor

# 初始化Agent执行器
agent_executor = init_bilibili_agent()

# ====================== 5. 核心调用函数（对外提供接口） ======================
def run_bilibili_agent(query: str) -> str:
    """调用B站运营Agent处理用户查询"""
    try:
        # 执行Agent并返回结果
        response = agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"Agent处理失败：{str(e)}"

# 测试代码（直接运行该文件即可调试）
if __name__ == "__main__":
    print("=== B站运营Agent启动（输入q退出）===")
    while True:
        user_query = input("请输入你的问题：")
        if user_query.lower() == "q":
            print("Agent已退出～")
            break
        result = run_bilibili_agent(user_query)
        print(f"\nAgent回复：{result}\n" + "-"*50)