# 博主智策：B站内容分析与 AI 运营助手 (RAG + Agentic Workflow)

本项目是一个工业级的 B 站自媒体运营辅助系统。它完整跑通了 **“数据采集 → 特征工程 → 机器学习预测 → 知识库构建 (RAG) → 智能体交互 (Agent)”** 的全链路落地流程。

## 🌟 你能得到什么？

- **🤖 交互式 AI 运营助手**：基于 LangChain 和 Streamlit 构建的 Web Agent，支持多轮对话、动态工具调度（Tool Calling），为你提供个性化的标题优化与流量诊断。
- **📊 混合检索增强 (RAG)**：摒弃传统词频匹配，采用 ChromaDB (Dense) + TF-IDF (Sparse) 双路召回引擎，并结合 Query Rewriting 与 HyDE 技术，精准挖掘历史爆款对标视频。
- **📈 科学的流量预测模型**：非黑盒模型。结合岭回归 (Ridge) 与逻辑回归 (Logistic)，不仅能预测播放量分档，更能输出基于特征权重的结构化可解释报告 (XAI)。
- **🧠 自动化数据洞察**：基于 BERT 向量化与 K-Means 自适应聚类，每天自动挖掘 B 站热门话题簇，并通过 Plotly 生成高维数据交互式大盘。
- **🛠️ 企业级工程架构**：严格的 MVC 分层设计，内置 Pydantic 数据校验，支持 Docker-Compose 一键微服务化部署，并集成 Prometheus + Grafana 全链路监控。

## 项目结构（重构版）

本项目已采用**分层架构**进行重构，以提高代码的可维护性、复用性和扩展性。

```
.
├── app/
│   ├── api/                 # API 路由定义 (接口层)
│   │   └── routes.py        # 定义 /api/topics, /api/predict_view 等路由
│   ├── core/                # 核心业务逻辑 (业务层)
│   │   ├── agent.py         # LangChain Agent 智能体核心逻辑 (工具调度与记忆)
│   │   ├── analysis.py      # 聚类结果查询与摘要逻辑
│   │   ├── predictor.py     # 播放量预测与分档模型推理
│   │   ├── recommender.py   # RAG 混合检索 (ChromaDB 稠密 + TF-IDF 稀疏 + RRF)
│   │   ├── scheduler.py     # 定时任务调度器 (APScheduler)
│   │   ├── data_loader.py   # 数据加载与缓存 (Singleton)
│   │   ├── config.py        # 统一路径与环境配置
│   │   └── llm.py           # LLM 接口及 Query Rewriting/HyDE 增强
│   ├── models/              # 数据模型定义 (数据层)
│   │   └── schemas.py       # Pydantic 模型 (Request/Response 校验)
│   ├── utils/               # 通用工具函数 (基础层)
│   │   ├── text_utils.py    # 分词、停用词、BERT 预处理
│   │   └── feature_utils.py # 标题特征工程、时间解析
│   ├── static/              # 静态资源
│   │   └── images/          # 可视化图表 (tsne.png, importance.png)
│   ├── templates/           # 前端模板
│   │   └── index.html       # 聚类与预测展示页面
│   └── __init__.py          # Flask 应用工厂
├── chroma_db/               # 本地向量数据库 (Chroma) 存储目录
├── data/                    # 数据存储
│   ├── raw/                 # 原始数据 (output.json)
│   └── processed/           # 处理后的结构化数据 (CSV)
├── models/                  # 模型存储
│   └── trained/             # 训练好的模型 (.pkl, .joblib, .npy)
├── scripts/                 # 离线任务脚本
│   ├── db_setup.py          # 数据库初始化 (建表)
│   ├── load_data.py         # 加载 CSV 数据到 DataFrame
│   ├── load_data_to_db.py   # 数据入库 (JSON -> MySQL)
│   ├── regenerate_importance_plot.py # 重新生成特征重要性图表 (RandomForest)
│   ├── run_eda.py           # 运行探索性数据分析 (EDA) 并生成图表
│   ├── topic_clustering.py  # 聚类分析脚本 (BERT 向量化 + ChromaDB入库 + K-Means)
│   ├── train_view_predictor.py # 训练播放量预测模型 (Ridge + Logistic)
│   └── evaluate_rag.py      # RAG 检索与生成质量评估 (RAGAS-style)
├── src/bilibili_scraper/    # Scrapy 爬虫项目
│   ├── bilibili_scraper/
│   │   ├── middlewares.py   # 随机 UA、代理中间件
│   │   ├── pipelines.py     # 数据校验、增量写入
│   │   ├── settings.py      # 反爬策略配置
│   │   └── spiders/video.py # 增量爬虫
├── app.py                   # Flask Web 服务启动入口
├── web_agent.py             # Streamlit 智能体交互界面启动入口
├── docker-compose.yml       # 容器化编排配置文件
├── Dockerfile               # 容器镜像构建文件
├── openapi.yaml             # Coze/插件对接 API 定义
├── requirements.txt         # 项目依赖
├── prompts/                 # Prompt 模板目录（与代码解耦）
│   ├── agent_system.txt     # Agent 系统 prompt
│   ├── title_optimize.txt   # 标题优化 prompt
│   ├── llm_analyze_title.txt # LLM 深度分析 prompt
│   ├── query_rewrite.txt    # Query 改写 prompt
│   ├── hyde_doc.txt         # HyDE 假设文档生成 prompt
│   └── loader.py            # Prompt 统一加载器
├── tests/                   # 单元测试目录
│   ├── test_schemas.py      # Pydantic 模型校验测试
│   ├── test_feature_utils.py # 特征工程函数测试
│   ├── test_predictor.py    # 预测模型测试（mock）
│   └── test_routes.py       # Flask API 路由测试
├── promptfooconfig.yaml     # Promptfoo 评测配置
└── README.md
```
 技术栈

  ┌────────────────┬───────────────────────────────────────────────────────┐
  │      层次      │                         技术                          │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ Web 框架       │ Flask 3.1 + Gunicorn（生产）                          │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ AI Agent       │ LangChain + 阿里云 DashScope（Qwen 系列 LLM）         │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ RAG / 向量检索 │ ChromaDB（稠密）+ TF-IDF（稀疏）+ RRF 融合            │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ Embedding      │ HuggingFace Transformers（BERT）                      │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ 机器学习       │ scikit-learn（Ridge / Logistic 回归）+ SHAP（XAI）    │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ 数据库         │ MySQL（业务数据）+ ChromaDB（向量）                   │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ 爬虫           │ Scrapy + BeautifulSoup4                               │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ 定时任务       │ APScheduler                                           │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ 前端           │ Jinja2 模板 + Plotly 交互图表 + Streamlit（Agent UI） │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ 监控           │ Prometheus + Grafana                                  │
  ├────────────────┼───────────────────────────────────────────────────────┤
  │ 部署           │ Docker Compose                                        │
  └────────────────┴───────────────────────────────────────────────────────┘

  ---
  入口文件

  ┌─────────────────┬─────────────────────────────────────────────────────────────┐
  │      文件       │                            用途                             │
  ├─────────────────┼─────────────────────────────────────────────────────────────┤
  │ app.py          │ Flask Web 服务入口，调用 create_app()                       │
  ├─────────────────┼─────────────────────────────────────────────────────────────┤
  │ web_agent.py    │ Streamlit AI 对话界面入口（streamlit run web_agent.py）     │
  ├─────────────────┼─────────────────────────────────────────────────────────────┤
  │ app/__init__.py │ Flask 工厂函数，注册蓝图、初始化模型/数据/调度器/Prometheus │
  └─────────────────┴─────────────────────────────────────────────────────────────┘

  ---
  主要模块（app/ 下）

  app/
  ├── api/routes.py        # 路由层：/api/predict_view, /api/title_advice 等
  ├── core/
  │   ├── agent.py         # LangChain Agent，Tool Calling + 滑动窗口记忆
  │   ├── recommender.py   # 混合 RAG 检索（Dense + Sparse + RRF）
  │   ├── predictor.py     # Ridge 回归预测播放量 + Logistic 分档
  │   ├── llm.py           # LLM 接口 + Query Rewriting + HyDE 增强
  │   ├── analysis.py      # K-Means 聚类结果查询与摘要
  │   ├── scheduler.py     # APScheduler 定时更新聚类
  │   ├── data_loader.py   # 单例数据加载与缓存
  │   └── config.py        # 统一路径与环境变量
  ├── models/schemas.py    # Pydantic 请求/响应校验
  └── utils/
      ├── text_utils.py    # jieba 分词、停用词、BERT 预处理
      └── feature_utils.py # 标题特征工程、时间段解析



## 核心改进

### 1. RAG 架构与多路召回引擎 (RAG & Hybrid Retrieval)
- **离线向量化入库 (Offline Vectorization)**：在数据处理流水线中引入 `transformers`，将清洗后的 B 站标题转化为高维稠密向量（BERT），并无缝写入 **ChromaDB** 向量数据库，构建高可用的离线特征集。
- **混合检索策略 (Hybrid Retrieval)**：摒弃了单一的字面匹配，采用 **Dense (BERT + ChromaDB 语义检索)** 与 **Sparse (TF-IDF 关键词检索)** 的双路召回架构。结合 **RRF (倒数秩融合)** 算法重新打分，既保证了长尾冷门词的精确召回，又具备极强的语义泛化能力。
- **检索前查询增强 (Query Enhancement)**：针对用户输入往往过于简短的问题，在 RAG 检索前置入 **Query Rewriting (大模型查询改写)** 和 **HyDE (假设性文档生成)** 技术。通过大模型生成“理想爆款标题”和“同义搜索词”去扩充检索空间，显著提升了相似视频的召回质量。

### 2. 智能体工作流与交互 (Agentic Workflow & UI)
- **Tool Calling 智能调度 (LangChain Agent)**：基于 **LangChain** 框架构建了核心运营 Agent。模型能够根据用户的自然语言意图，自主决策并调度不同的底层能力（如：触发闲聊引擎、调用预测回归模型、执行 RAG 检索与改写）。
- **滑动窗口记忆机制 (Conversation Memory)**：为 Agent 注入 `ConversationBufferWindowMemory`，使其具备处理多轮复杂对话的能力。Agent 能够结合历史上下文进行精准微调。
- **现代化交互界面 (Streamlit)**：摒弃了对非技术人员不友好的终端命令行，使用 **Streamlit** 构建了流畅的 ChatGPT 式 Web 交互界面，支持实时流式输出和状态提示，极大提升了产品的可用性和商业演示效果。

### 3. 数据流底座与爬虫工程 (Data Pipeline & Scrapy)
- **反爬与高可用策略**：基于 Scrapy 框架开发了工业级爬虫，内置随机 User-Agent 轮换、智能延时抖动以及自动重试机制（拦截 403/429/5xx 状态码），保障了数据采集的高可用性。
- **增量同步 (Incremental Crawling)**：支持通过指令（`-a incremental=True`）跳过数据库中已存在的 `video_id`，大幅降低带宽消耗与 B 站服务器压力。
- **数据清洗与防御性编程**：在 Pipeline 中引入严格的数据校验机制，自动修正异常数值类型，丢弃脏数据，确保进入下游模型的都是高质量语料。

### 4. 机器学习模型与可解释性 (Predictive Modeling & XAI)
- **自适应聚类分析 (Auto-K Clustering)**：自动化探索最优聚类参数，通过计算**轮廓系数 (Silhouette Score)** 在 `K=5~15` 区间内动态寻优。结合 PCA 与 t-SNE 降维技术，利用 Plotly 生成高维数据的交互式可视化散点图。
- **多特征融合预测 (Ridge & Logistic Regression)**：除了标题文本特征外，创新性地引入了“UP主历史数据均值”、“发布时间时段”以及“分区流量天花板”等结构化因子，构建了播放量连续值预测（回归）和爆款分档（分类）双模型。
- **模型可解释性 (XAI)**：打破“黑盒”预测，模型在输出预估播放量的同时，能够基于特征工程的权重系数（如：标题长度、疑问句式、高优分区），输出结构化的分析理由，增强业务人员对预测结果的信任度。

### 5. 工程化规范与云原生部署 (DevOps & Best Practices)
- **领域驱动的分层架构**：将单体 Flask 应用重构为清晰的四层架构：`api`（路由层）、`core`（业务逻辑层）、`models`（数据模式层）、`utils`（基础工具层），实现高内聚低耦合。
- **Pydantic 严格校验**：在 API 层面全面接入 Pydantic Model，对前后端交互数据进行强类型约束，提升接口健壮性。
- **微服务容器化 (Docker Compose)**：编写了符合最佳实践的 `Dockerfile`，并使用 `docker-compose` 一键拉起 Web 服务、MySQL 数据库以及监控套件，实现了环境隔离与极速部署。
- **可观测性监控体系 (Observability)**：无缝集成 **Prometheus** 暴露应用级指标，配合 **Grafana** 可视化大盘，实时监控核心 API 的 QPS、响应延迟及错误率，满足生产级运维需求。

## 🚀 快速开始与运行指南

### 0. 先决条件与环境安装

- Python 3.9+（建议 3.10/3.11）
- MySQL 8.0+（如果需要跑离线数据处理流程）
- 阿里云 DashScope API Key（用于启动 Agent）

```powershell
# 1. 克隆项目并进入目录
cd f:\就业\项目\博主项目

# 2. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 3. 安装核心依赖
pip install -r requirements.txt

# 4. 配置环境变量 (请替换为你的真实 Key)
$env:DASHSCOPE_API_KEY="sk-xxxxxx"
$env:DB_PASSWORD="你的数据库密码"
```

### 1. 体验核心功能：启动 Agent 智能体交互界面
这是本项目对最终用户最友好的入口。我们提供了一个基于 Streamlit 构建的 ChatGPT 风格 Web 界面。

**启动命令：**
```powershell
streamlit run web_agent.py
```
**你能得到什么：** 
浏览器会自动打开 `http://localhost:8501`。
**你可以干什么：**
- 直接向 Agent 提问：“帮我预测一下这个标题的播放量：Python爬虫3天速成”。
- 要求 Agent 进行诊断：“帮我优化这个标题：去三亚旅游的vlog，要求侧重穷游党”。
- Agent 会自动思考，调用底层的机器学习模型或 RAG 检索系统，并用自然语言将专业建议反馈给你。

### 2. 开发者后台：启动 Flask Web API & 监控大盘
如果你是前端开发人员或需要直接调用接口，可以启动 Flask 后端。

**启动命令：**
```powershell
python app.py
```
**你能得到什么：**
- 浏览器访问 `http://127.0.0.1:5000` 可以看到 B 站热门话题聚类（K-Means/DBSCAN）的 3D 散点图和数据大盘。
- 提供了 `/api/predict_view` 和 `/api/title_advice` 等 RESTful API 供前端或第三方应用（如飞书机器人、Coze 插件）调用。
- 后台会自动启动 `APScheduler`，每天凌晨自动更新聚类数据。

### 3. 数据工程师：离线流水线与 RAG 评测
如果你想从零开始构建数据，或者重新训练模型，请按顺序执行以下离线脚本：

**步骤 A：数据采集**
```powershell
cd src\bilibili_scraper
scrapy crawl video -a target_count=1000 -O output.json # 全量抓取
scrapy crawl video -a incremental=True -a target_count=1000 # 增量抓取
```
*产出：最新 B 站热门视频元数据（JSON 格式）。*

**步骤 B：特征提取与模型训练**
```powershell
# 1. BERT 聚类分析与 ChromaDB 向量入库
python scripts/topic_clustering.py

# 2. 训练播放量预测双模型 (回归+分类)
python scripts/train_view_predictor.py
```
*产出：更新 `models/trained/` 目录下的 `.joblib` 模型权重文件，并生成最新的 `chroma_db/` 本地向量库供在线 RAG 检索使用。*

**步骤 C：自动化质量评估**
```powershell
# 运行 RAG 系统的 Recall 指标与 LLM-as-a-Judge 评估
python scripts/evaluate_rag.py
```
*产出：在终端输出当前检索系统的 Recall@K 命中率，以及大模型生成的诊断建议的质量打分（1-5分）。用于验证你修改 Prompt 或更换 Embedding 模型后，系统效果是变好了还是变差了。*

### 4. 生产级部署：Docker 容器化架构
如果你想一键启动包含 数据库、Web服务、监控体系的完整微服务架构：

```powershell
# 修改 docker-compose.yml 中的环境变量后执行：
docker-compose up -d --build
```
**架构全貌：**
- **业务服务**: `http://localhost:5000`

*(如需清理环境，可执行：`docker-compose down --rmi all`)*

## 📚 接口文档与外网暴露

本项目的接口完全符合 OpenAPI 3.0 规范，详细定义见根目录的 [`openapi.yaml`](file:///f:/就业/项目/博主项目/openapi.yaml)。

如果需要将本地服务暴露给外部（例如对接字节 Coze 插件），可以使用 Cloudflare Tunnel：
```powershell
cd F:\cloudflared
.\cloudflared.exe tunnel --url http://localhost:5000
```

---

## 🔧 2026年3月重构说明

本次重构以**「以可测试性为核心驱动力」**为主线，在不改变任何业务功能的前提下，修复了阻碍工程化落地的五类根因问题。

### 1. Docker 基础设施修复

**根因：** 容器启动时 `web` 服务早于 MySQL 就绪，导致 Flask 首次连接数据库必然失败（race condition）。

**改动清单：**

| 文件 | 改动 | 解决问题 |
|---|---|---|
| `docker-compose.yml` | 为 `db` 服务添加 `healthcheck`（mysqladmin ping，5次重试，30s启动宽限期） | 消除启动竞态 |
| `docker-compose.yml` | `web` 的 `depends_on` 改为 `condition: service_healthy` | 保证 DB 就绪才启动 Web |
| `docker-compose.yml` | 挂载 `./data`, `./models`, `./chroma_db` 到容器 | 模型文件不再需要打入镜像 |
| `scripts/init_db.sql` | 从空目录改为真实 SQL 文件（含 `bilibili_data` 库和 `videos` 表 DDL） | 修复 MySQL init 挂载报错 |

**架构决策：** `service_healthy` 是 Docker Compose 原生的依赖就绪方案，比 `sleep` 或 entrypoint 重试脚本更可靠，且不需要在应用层增加连接重试逻辑。

---

### 2. 消除模块级副作用（Module-Level Side Effects）

**根因：** 多个核心模块在 `import` 时立即执行网络请求、模型加载、环境变量写入，导致单测环境无法 `import` 这些模块（无 API Key 或网络时直接抛异常）。

**改动清单：**

| 文件 | 原有副作用 | 修复方式 |
|---|---|---|
| `app/core/agent.py` | 模块顶层实例化 `ChatOpenAI`、`AgentExecutor`、`Chroma` | 全部移入 `init_bilibili_agent()` 函数，首次调用时才执行 |
| `app/core/agent.py` | `sys.path.append` hack | 删除，正确通过 `app.py` 启动时 Python 包查找机制自动处理 |
| `app/core/recommender.py` | `os.environ['HF_ENDPOINT']` 在顶层赋值 | 移入 `init_recommender()` 内部 |
| `app/core/scheduler.py` | `logging.basicConfig()` 在顶层调用 | 删除，logging 配置权属于应用工厂 `app/__init__.py` |

**架构决策：** 模块只声明意图（哨兵变量），不执行初始化。谁负责启动（应用工厂），谁负责调用 `init_*()` 函数。

---

### 3. 统一懒加载模式（Lazy Initialization Pattern）

**根因：** 各模块初始化逻辑散乱，部分模块在业务函数内部隐式触发加载，导致初始化时机不可预测。

**统一模式：** 所有重资源模块均采用 `哨兵变量 + 显式 init 函数 + 应用工厂统一调用` 三件套：

```python
# 示例（以 predictor.py 为代表）
_VIEW_PREDICTOR = None          # 哨兵变量，模块级
_VIEW_BUCKET_CLASSIFIER = None

def load_models():               # 显式初始化函数
    global _VIEW_PREDICTOR, _VIEW_BUCKET_CLASSIFIER
    _VIEW_PREDICTOR = joblib.load(...)
    _VIEW_BUCKET_CLASSIFIER = joblib.load(...)

def predict_view(req):           # 业务函数：不自动 init，模型未加载直接报错
    if _VIEW_PREDICTOR is None:
        raise RuntimeError("模型未加载，请先调用 load_models()")
    ...
```

**架构决策（关键）：** `predict_view` 选择抛出 `RuntimeError` 而非静默地调用 `load_models()`。这是故意的——让配置错误在启动时暴露，而不是在第一个真实请求时才暴露（fail-fast 原则）。

| 模块 | 哨兵变量 | Init 函数 |
|---|---|---|
| `agent.py` | `_agent_executor` | `init_bilibili_agent()` |
| `recommender.py` | `_CHROMA_CLIENT` 等 8 个 | `init_recommender()` |
| `predictor.py` | `_VIEW_PREDICTOR` 等 2 个 | `load_models()` |
| `data_loader.py` | `_DF` | `load_data()` |

---

### 4. Prompt 与代码解耦

**根因：** 所有 Prompt 模板硬编码在业务逻辑函数中，修改 Prompt 必须改 Python 文件，且无法用自动化工具评测质量。

**改动清单：**

| 新增文件 | 内容 |
|---|---|
| `prompts/agent_system.txt` | Agent 系统 Prompt 模板 |
| `prompts/title_optimize.txt` | 标题优化 Prompt（含 `{docs}`, `{query}` 变量） |
| `prompts/llm_analyze_title.txt` | 标题深度分析 Prompt（含6个变量） |
| `prompts/query_rewrite.txt` | Query 改写 Prompt（返回 JSON 数组） |
| `prompts/hyde_doc.txt` | HyDE 假设文档生成 Prompt |
| `prompts/loader.py` | 统一加载器，带内存缓存，避免重复 IO |

**架构决策：** Prompt 文件是**独立可评测的工件**，不是代码。分离后：
- 运营/产品可直接修改 `.txt` 文件迭代 Prompt，无需接触 Python
- `promptfooconfig.yaml` 可对每个 Prompt 独立做 A/B 评测（`npx promptfoo eval`）
- 变更历史通过 git blame 单独追踪

---

### 5. 测试体系建设

**根因：** 原项目零测试覆盖，前四项重构的正确性无法验证。

**新增测试文件：**

| 文件 | 测试对象 | 关键技术 |
|---|---|---|
| `tests/test_schemas.py` | Pydantic `PredictRequest` / `PredictViewResponse` | `pytest.raises(ValidationError)` |
| `tests/test_feature_utils.py` | `extract_features()` 特征工程函数 | 纯函数测试，无依赖 |
| `tests/test_predictor.py` | `predict_view()` 预测逻辑 | `unittest.mock.patch` 注入假模型 |
| `tests/test_routes.py` | `/api/predict_view`, `/api/title_advice` 路由 | Flask `test_client()` + patch 服务层 |

**关键洞察：** 测试套件之所以能写出来，**直接受益于前三项重构**。在重构前，`import agent` 会触发 API 连接，`import recommender` 会下载 BERT 模型，任何测试初始化都会失败。重构后，`patch` 哨兵变量即可完全隔离外部依赖。

**运行方式：**
```powershell
pip install pytest pytest-mock
pytest tests/ -v
```

---
# 直接运行 startup.py 启动应用
python startup.py直接自动化开启