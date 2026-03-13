# 博主智策：B站热门内容分析与选题辅助（Flask + Scrapy + BERT 聚类）

这个仓库把“爬热门数据 → 入库 → 聚类分析 → Web/API 展示 →（可选）预测播放量/标题建议”这一整条链路做成了可跑通的工程。

## 你能得到什么

- 热门内容聚类：对标题做 BERT 向量化，用 K-Means/DBSCAN 聚类，输出每个话题簇的关键词、Top 视频、播放量统计。
- 可视化产物：t-SNE 聚类散点图、特征重要性条形图（保存在 [app/static/images](file:///f:/就业/项目/博主项目/app/static/images)）。
- 选题辅助 API：预测播放量（回归）、预测分档（分类）、标题建议（LLM 智能体诊断 + 相似爆款 + 关键词）。
- 一个无框架前端页面：直接调用后端 API 展示话题簇（[index.html](file:///f:/就业/项目/博主项目/app/templates/index.html)）。

## 项目结构（工程化重构版）

本项目已采用**分层架构**进行重构，以提高代码的可维护性、复用性和扩展性。

```
.
├── app/
│   ├── api/                 # API 路由定义 (接口层)
│   │   └── routes.py        # 定义 /api/topics, /api/predict_view 等路由
│   ├── core/                # 核心业务逻辑 (业务层)
│   │   ├── analysis.py      # 聚类结果查询与摘要逻辑
│   │   ├── predictor.py     # 播放量预测与分档模型推理
│   │   ├── recommender.py   # 标题相似度推荐 (TF-IDF)
│   │   ├── scheduler.py     # 定时任务调度器 (APScheduler)
│   │   ├── data_loader.py   # 数据加载与缓存 (Singleton)
│   │   ├── config.py        # 统一路径与环境配置
│   │   └── llm.py           # LLM 智能体接口 (DashScope/Qwen)
│   ├── models/              # 数据模型定义 (数据层)
│   │   └── schemas.py       # Pydantic 模型 (Request/Response 校验)
│   ├── utils/               # 通用工具函数 (基础层)
│   │   ├── text_utils.py    # 分词、停用词、BERT 预处理、TF-IDF 关键词
│   │   └── feature_utils.py # 标题特征工程、时间解析
│   ├── static/              # 静态资源
│   │   └── images/          # 可视化图表 (tsne.png, importance.png)
│   ├── templates/           # 前端模板
│   │   └── index.html       # 聚类展示页面
│   ├── __init__.py          # Flask 应用工厂
│   └── app.py               # (废弃/重构) 仅作为启动入口
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
│   ├── topic_clustering.py  # 聚类分析脚本 (BERT + K-Means/DBSCAN)
│   └── train_view_predictor.py # 训练播放量预测模型 (Ridge + Logistic)
├── src/bilibili_scraper/    # Scrapy 爬虫项目
│   ├── bilibili_scraper/
│   │   ├── middlewares.py   # 随机 UA、代理中间件
│   │   ├── pipelines.py     # 数据校验、增量写入
│   │   ├── settings.py      # 反爬策略配置
│   │   └── spiders/video.py # 增量爬虫
├── openapi.yaml             # Coze/插件对接 API 定义
├── requirements.txt         # 项目依赖
└── README.md
```

## 核心改进

### 1. 架构重构 (Architecture Refactoring)
- **分层设计**：将单文件 Flask 应用拆分为 `api` (路由), `core` (业务), `models` (数据), `utils` (工具) 四层，解耦了 HTTP 处理与核心算法。
- **配置管理**：统一通过 `app.core.config` 管理路径与环境变量，消除了硬编码路径。
- **数据流规范**：明确了 `data/raw` (原始), `data/processed` (清洗后), `models/trained` (模型) 的数据流向。

### 2. 爬虫升级 (Robust Crawler)
- **反爬策略**：
    - **随机 User-Agent**：中间件每次请求随机切换浏览器指纹。
    - **智能延迟**：`DOWNLOAD_DELAY=2` 配合随机抖动，模拟真人行为。
    - **自动重试**：配置 `RETRY_TIMES=3`，自动处理 403/429/5xx 错误。
- **增量抓取 (Incremental Crawling)**：
    - 支持 `-a incremental=True` 参数，启动时自动加载已有 ID，跳过重复数据，极大降低对 B 站服务器的压力。
- **数据校验 (Data Validation)**：
    - 引入 `BilibiliDataValidationPipeline`，强制校验关键字段（`video_id`, `title`），自动修正数值类型，丢弃脏数据。

### 3. 聚类分析优化 (Advanced Clustering)
- **Auto-K Selection**：不再写死 `k=10`，而是通过计算 **轮廓系数 (Silhouette Score)** 自动寻找 `k=5~15` 范围内的最优聚类数。
- **语义关键词 (Semantic Keywords)**：使用 **TF-IDF** 算法替代简单的词频统计，自动过滤通用停用词，提取更具代表性的簇标签。
- **交互式可视化 (Interactive Viz)**：引入 **Plotly** 生成交互式 HTML 散点图，支持缩放、悬停查看视频详情，替代静态 PNG 图片。
- **自动化闭环 (Automation)**：集成 **APScheduler**，每日凌晨 2 点自动运行聚类脚本，实现数据分析的自动更新。

### 4. 预测模型与建议优化 (Predictive Modeling & Advice)
- **科学评估 (Rigorous Evaluation)**：引入 `train_test_split`，计算并输出 **MAE**、**RMSE**、**R2** 等多维度指标，并持久化保存评估报告。
- **可解释性 (XAI)**：新增 **“分区定位”影响因子**（如知识区 vs 生活区的流量差异），并移除了不准确的“情绪词”分析，使预测解释更具针对性。
- **智能体诊断 (Agentic Workflow)**：升级 `/api/title_advice`，引入 **LLM 智能体 (Qwen-Plus)**。不仅提供规则建议，还结合 **Random Forest 全局特征重要性** 和 **相似爆款**，生成深度标题诊断报告与差异化优化方案。

### 5. 工程化实践 (Engineering Practices)
- **Pydantic 校验**：API 入参使用 Pydantic Model 进行严格校验，提升接口健壮性。
- **复用性提升**：特征提取逻辑 (`app.utils.feature_utils`) 被训练脚本和在线预测 API 共享，避免 Training-Serving Skew。
- **容器化部署 (Docker)**：提供 `Dockerfile` 和 `docker-compose.yml`，一键拉起 Flask + MySQL + Prometheus + Grafana。
- **可观测性 (Observability)**：集成 **Prometheus** 监控指标，暴露 `/metrics` 端点，实时监控 QPS 和响应延迟。

## 先决条件

- Python 3.9+（建议 3.10/3.11）
-（可选）MySQL 8.0+：只在你要“重新入库/重新聚类”时需要

## 安装

```powershell
cd f:\就业\项目\博主项目
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行指南

### 1. 启动 Web 服务 (API & Dashboard)

Web 服务启动时会自动初始化调度器，每日定时更新聚类结果。

```powershell
python app.py
```
访问 http://127.0.0.1:5000 查看聚类结果或调用 API。

### 2. 运行爬虫 (支持增量)

```powershell
cd src\bilibili_scraper
# 首次全量抓取
scrapy crawl video -a target_count=1000 -O output.json

# 后续增量抓取 (跳过已抓取的 ID)
scrapy crawl video -a incremental=True -a target_count=1000
```
注意：Scrapy 命令必须在 `src/bilibili_scraper` 目录下运行。

### 3. 数据处理与建模

脚本运行需要设置数据库密码（若未写入环境变量）。

```powershell
# 临时设置环境变量 (仅当前会话有效)
$env:DB_PASSWORD = "你的数据库密码"

# 1. 聚类分析 (依赖 MySQL)
python scripts/topic_clustering.py

# 2. 训练预测模型 (依赖 MySQL)
python scripts/train_view_predictor.py
```

### 4. 容器化部署 (Docker Compose)

如果你想一键启动整个微服务架构（包含 MySQL、Prometheus 监控）：

```powershell
# 修改 docker-compose.yml 中的数据库密码
docker-compose up -d --build
```
- Web 应用: http://localhost:5000
- Grafana 监控: http://localhost:3000 (默认 admin/admin)
- Prometheus: http://localhost:9090

**注意**：如果你想在测试完后释放空间，请运行以下命令清理所有容器和镜像：
```powershell
docker-compose down --rmi all
docker system prune -f
```


## API 使用

接口定义见 [openapi.yaml](file:///f:/就业/项目/博主项目/openapi.yaml)。

- GET /api/topics：话题簇列表
- POST /api/predict_view：预测播放量 (含 `explanations` 解释)
- POST /api/title_advice：标题建议 (含 `advice_list` 优化建议)

PowerShell 示例：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/title_advice `
  -ContentType "application/json" `
  -Body (@{ title="新手入门：3天做出爆款视频？"; category="知识" } | ConvertTo-Json)
```
# 内网穿透
cd F:\cloudflared
.\cloudflared.exe tunnel --url http://localhost:5000
