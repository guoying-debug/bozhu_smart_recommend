# 博主智策：B站热门内容分析与选题辅助（Flask + Scrapy + BERT 聚类）（代码在master分支里面）

这个仓库把“爬热门数据 → 入库 → 聚类分析 → Web/API 展示 →（可选）预测播放量/标题建议”这一整条链路做成了可跑通的工程。

## 你能得到什么

- 热门内容聚类：对标题做 BERT 向量化，用 K-Means/DBSCAN 聚类，输出每个话题簇的关键词、Top 视频、播放量统计。
- 可视化产物：t-SNE 聚类散点图、特征重要性条形图（保存在 [models](file:///f:/就业/项目/博主项目/models) 与 [static](file:///f:/就业/项目/博主项目/app/static)）。
- 选题辅助 API：预测播放量（回归）、预测分档（分类）、标题建议（相似爆款 + 关键词 + 类目推荐）。
- 一个无框架前端页面：直接调用后端 API 展示话题簇（[index.html](file:///f:/就业/项目/博主项目/app/templates/index.html)）。

## 项目结构（关键入口）

```
.
├── app/
│   ├── templates/index.html         # 页面：聚类结果展示（只用 /api/topics）
│   ├── static/                      # 图：kmeans_tsne_visualization.png / feature_importance.png
│   └── app.py                       # 服务入口：Flask API + 页面渲染
├── models/                          # 运行期依赖/产物：CSV、joblib、pkl、png、npy
├── scripts/
│   ├── db_setup.py                  # 建表：videos
│   ├── load_data_to_db.py           # output.json -> MySQL（会 TRUNCATE videos）
│   ├── topic_clustering.py          # 从 MySQL 读数据做 BERT 聚类，写 CSV/图/模型
│   └── train_view_predictor.py      # 训练播放量回归 + 分档分类，写 joblib
├── src/bilibili_scraper/            # Scrapy 爬虫项目（spider: video）
├── openapi.yaml                     # Coze/插件对接用的 OpenAPI 定义
├── requirements.txt
└── README.md
```

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

如果你要运行 BERT 聚类脚本（[topic_clustering.py](file:///f:/就业/项目/博主项目/scripts/topic_clustering.py)），还需要额外安装：

```powershell
pip install transformers torch
```

## 运行方式 1：只启动 Web/API（最省事）

只要 [models/videos_with_clusters_and_coords.csv](file:///f:/就业/项目/博主项目/models/videos_with_clusters_and_coords.csv) 存在，就可以直接启动服务，不强制依赖 MySQL。

```powershell
cd f:\就业\项目\博主项目
python app/app.py
```

浏览器打开：

- http://127.0.0.1:5000

启动时做了什么（对应 [app.py](file:///f:/就业/项目/博主项目/app/app.py)）：

- 读取 models/videos_with_clusters_and_coords.csv 到内存
- 计算聚类摘要（K-Means 各簇计数、DBSCAN 噪声比例等）
-（若存在）加载 models/view_predictor.joblib 与 models/view_bucket_classifier.joblib
- 构建标题相似度 TF-IDF 矩阵（用于 /api/title_advice 的相似爆款检索）

## 运行方式 2：从零跑全流程（重新抓取/入库/聚类/训练）

### 0) 准备 MySQL 与环境变量

先在 MySQL 里创建数据库（默认名 bilibili_data）：

```sql
CREATE DATABASE bilibili_data CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
```

Windows PowerShell 设置连接信息（只设置 DB_PASSWORD 也能跑默认参数）：

```powershell
$env:DB_USER="root"
$env:DB_PASSWORD="你的数据库密码"
$env:DB_HOST="localhost"
$env:DB_PORT="3306"
$env:DB_NAME="bilibili_data"
```

### 1) 建表

```powershell
python scripts/db_setup.py
```

### 2) 抓取热门数据到 output.json（覆盖写入）

爬虫入口在 [video.py](file:///f:/就业/项目/博主项目/src/bilibili_scraper/bilibili_scraper/spiders/video.py)，支持 `-a target_count=5000`。

```powershell
cd f:\就业\项目\博主项目\src\bilibili_scraper
scrapy crawl video -a target_count=5000 -O output.json
```

说明：

- `-O` 是覆盖写入，避免多次运行导致 JSON 格式混乱
- 每页 20 条，target_count=5000 大约 250 页；爬虫自带 2s 延迟，耗时 8~10 分钟属于正常范围

### 3) JSON 入库（会清空旧数据）

```powershell
cd f:\就业\项目\博主项目
python scripts/load_data_to_db.py
```

### 4) 重新聚类/生成可视化/导出 CSV

```powershell
python scripts/topic_clustering.py
```

会产出或更新：

- models/videos_with_clusters_and_coords.csv
- models/bert_embeddings.npy（如果存在且条数一致会复用）
- models/bert_kmeans_model.pkl、models/bert_dbscan_model.pkl
- models/kmeans_tsne_visualization.png、models/feature_importance.png（同时页面会从 app/static 读取同名图）

### 5)（可选）训练播放量预测与分档模型

```powershell
python scripts/train_view_predictor.py
```

会更新：

- models/view_predictor.joblib（回归：预测播放量）
- models/view_bucket_classifier.joblib（分类：预测分档）

### 6) 重启 Web 服务

```powershell
python app/app.py
```

## API 使用

接口定义见 [openapi.yaml](file:///f:/就业/项目/博主项目/openapi.yaml)，实际实现见 [app.py](file:///f:/就业/项目/博主项目/app/app.py)。

- GET /api/topics：话题簇列表（按总播放量降序）
- GET /api/analysis_summary：聚类摘要（K-Means 计数、DBSCAN 噪声比例等）
- POST /api/predict_view：预测播放量（需要 models/view_predictor.joblib）
- POST /api/predict_bucket：预测播放量分档（需要 models/view_bucket_classifier.joblib）
- POST /api/title_advice：标题建议（相似爆款/关键词/类目推荐；预测是可选的）

PowerShell 示例：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/topics

Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/title_advice `
  -ContentType "application/json" `
  -Body (@{ title="新手入门：3天做出爆款视频？"; category="知识" } | ConvertTo-Json)
```

## 常见问题

- 启动报 “找不到数据文件 … videos_with_clusters_and_coords.csv”：先跑 `python scripts/topic_clustering.py` 或确认 models/ 里已有该文件。
- `topic_clustering.py` 提示 DB_PASSWORD 未设置：该脚本强制从 MySQL 读数据，必须设置 DB_PASSWORD 且数据库可连。
- 爬虫抓取中断/返回空：先降低 target_count 做小规模验证；需要更稳可以进一步加重试、降频、代理等策略（入口在 [video.py](file:///f:/就业/项目/博主项目/src/bilibili_scraper/bilibili_scraper/spiders/video.py)）。

##  内网穿透到COZE的API
本地先跑起来 → 用穿透工具给它一个公网 HTTPS 地址 → 把这个公网地址写到 Coze 的 OpenAPI servers/baseUrl 里 。
1) 安装 cloudflared（Windows 最简单方式：直接下载 exe） 在 PowerShell 执行：

```
mkdir C:\Tools\cloudflared -Force | Out-Null
cd C:\Tools\cloudflared

# 64位 Windows（绝大多数都是这个）
Invoke-WebRequest -Uri "https://github.com/
cloudflare/cloudflared/releases/latest/download/
cloudflared-windows-amd64.exe" -OutFile "cloudflared.
exe"

# 验证
.\cloudflared.exe --version
```
2) 先启动你的项目服务（本地 5000 要先跑起来） 在项目根目录：

```
cd f:\就业\项目\博主项目
.\.venv\Scripts\python app\app.py
```
3) 再开一个 PowerShell 窗口做内网穿透

```
.\cloudflared.exe tunnel --url http://localhost:5000
```




1) 别关这个 cloudflared 窗口

- quick tunnel 必须保持这个进程一直运行；关了就失效、域名也可能换。
2) 确认你的 Flask 后端也在跑（本地 5000） 另开一个 PowerShell（别用正在跑 cloudflared 的那个窗口），在项目根目录启动：

```
cd f:\就业\项目\博主项目
.\.venv\Scripts\python app\app.py
```
3) 先用公网地址自测接口是否通

```
Invoke-RestMethod "https://
clark-assumption-established-mixture.trycloudflare.
com/api/topics"
```
- 能返回 JSON：穿透 OK



4) 更新 openapi.yaml 的 servers.url（给 Coze 用）
https://clark-assumption-established-mixture.trycloudflare.com
