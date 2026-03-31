import streamlit as st
import os
import sys

# 解决 ModuleNotFoundError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app.core.agent import run_bilibili_agent
from app.core.predictor import load_models
from app.core.data_loader import load_data

# 初始化：加载模型和数据（仅执行一次）
@st.cache_resource
def init_app():
    """应用启动时初始化所有资源"""
    load_data()      # 加载数据
    load_models()    # 加载预测模型
    return True

# 执行初始化
init_app()

# 页面配置
st.set_page_config(
    page_title="博主智策 - 运营智能助手",
    page_icon="🤖",
    layout="centered"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .main {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Bilibili_logo.svg/1024px-Bilibili_logo.svg.png", width=150)
    st.title("🤖 运营智能助手")
    st.markdown("""
    这是一个基于 **LangChain** + **DashScope (Qwen)** 构建的 B 站运营 Agent。
    
    它可以帮你：
    - ✨ **优化标题**：基于 RAG 混合检索真实爆款库，给出修改建议。
    - 📈 **预测播放量**：调用机器学习模型，预估播放量区间。
    - 💬 **日常闲聊**：简单的问候与交流。
    
    **提示**：尝试输入 *"帮我优化标题：Python入门教程"*
    """)
    
    st.divider()
    if st.button("🧹 清除对话历史"):
        st.session_state.messages = []
        st.rerun()

# 聊天主界面
st.title("💬 B站运营 Agent 聊天室")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是 B 站运营智能助手。你可以让我帮你优化标题，或者预测某个标题的播放量。请问有什么可以帮你的吗？"}
    ]

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你的问题，例如：帮我优化标题：Python从入门到放弃"):
    # 将用户输入添加到历史记录并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示 Agent 思考状态
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考并调用工具..."):
            # 调用 Agent 核心逻辑
            response = run_bilibili_agent(prompt)
            st.markdown(response)
            
    # 将 Agent 回复添加到历史记录
    st.session_state.messages.append({"role": "assistant", "content": response})
