# Scrapy settings for bilibili_scraper project

BOT_NAME = "bilibili_scraper"

SPIDER_MODULES = ["bilibili_scraper.spiders"]
NEWSPIDER_MODULE = "bilibili_scraper.spiders"

# --- 反爬策略 ---
# 1. 不遵守 robots.txt (B站会禁止爬虫)
ROBOTSTXT_OBEY = False

# 2. 下载延迟 (降低频率)
DOWNLOAD_DELAY = 2  # 每次请求间隔 2 秒
RANDOMIZE_DOWNLOAD_DELAY = True # 随机化延迟 (0.5 * DELAY ~ 1.5 * DELAY)

# 3. 禁用 Cookies (防止被追踪)
COOKIES_ENABLED = False

# 4. 并发设置
CONCURRENT_REQUESTS = 4 # 降低并发数
CONCURRENT_REQUESTS_PER_DOMAIN = 4

# 5. 重试机制 (网络波动/封禁时重试)
RETRY_ENABLED = True
RETRY_TIMES = 3  # 重试 3 次
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429, 403] # 遇到这些码重试

# 6. 中间件配置 (随机 UA)
DOWNLOADER_MIDDLEWARES = {
    'bilibili_scraper.middlewares.RandomUserAgentMiddleware': 400,
    # 'bilibili_scraper.middlewares.ProxyMiddleware': 410, # 如果有代理池，解除注释
    'bilibili_scraper.middlewares.BilibiliScraperDownloaderMiddleware': 543,
}

# 7. 数据管道 (数据校验与入库)
ITEM_PIPELINES = {
    'bilibili_scraper.pipelines.BilibiliDataValidationPipeline': 300,
    'bilibili_scraper.pipelines.BilibiliJsonWriterPipeline': 350,
    # 'bilibili_scraper.pipelines.BilibiliMysqlPipeline': 400, # 如果需要直接入库，解除注释
}

# 8. User-Agent 池
USER_AGENT_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
]

# 日志级别
LOG_LEVEL = 'INFO'

# 编码
FEED_EXPORT_ENCODING = "utf-8"
