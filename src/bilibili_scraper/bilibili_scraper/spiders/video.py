import scrapy
import json
import logging
import math
from bilibili_scraper.items import BilibiliScraperItem

class VideoSpider(scrapy.Spider):
    # 爬虫的唯一标识名
    name = "video"
    # 允许爬取的域名
    allowed_domains = ["api.bilibili.com"]
    
    # 为这个爬虫单独设置的配置
    custom_settings = {
        'CONCURRENT_REQUESTS': 1,  # 并发请求数，设置为1，降低请求频率
        'DOWNLOAD_DELAY': 2,       # 下载延迟，设置为2秒，避免对服务器造成太大压力
    }

    def __init__(self, target_count=5000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_size = 20
        self.target_count = int(target_count)
        self.max_pages = max(1, int(math.ceil(self.target_count / self.page_size)))
        self.crawled_count = 0

    def start_requests(self):
        """
        这是爬虫的入口函数，Scrapy会从这里开始发起第一个请求。
        """
        # 伪造请求头，模拟浏览器访问
        headers = {
            'Referer': 'https://www.bilibili.com/v/popular/rank/all',
        }
        # 构造初始请求URL，ps=20表示每页20条数据，pn=1表示第一页
        yield scrapy.Request(
            url="https://api.bilibili.com/x/web-interface/popular?ps=20&pn=1",
            callback=self.parse,  # 指定处理响应的函数
            meta={'page_number': 1},  # 传递额外数据，这里用来记录页码
            headers=headers
        )

    def parse(self, response):
        """
        这是处理服务器响应的核心函数。
        它会解析返回的JSON数据，提取视频信息，并自动翻页。
        """
        page_number = response.meta['page_number']
        logging.info(f"--- 正在解析第 {page_number} 页 ---")
        
        # 将响应体（JSON字符串）解析为Python字典
        data = json.loads(response.body)
        
        # 检查API返回的数据是否正常
        if data.get('code') == 0 and data.get('data', {}).get('list'):
            logging.info(f"成功获取第 {page_number} 页的数据。项目数量: {len(data['data']['list'])}")
            
            # 遍历当前页的每一个视频数据
            for video_data in data['data']['list']:
                if self.crawled_count >= self.target_count:
                    break
                item = BilibiliScraperItem()
                # 从API返回的数据中提取我们关心的字段
                item['video_id'] = video_data.get('bvid')
                item['title'] = video_data.get('title')
                item['description'] = video_data.get('desc')
                item['author'] = video_data.get('owner', {}).get('name')
                item['author_id'] = video_data.get('owner', {}).get('mid')
                item['publish_time'] = video_data.get('pubdate')
                item['view_count'] = video_data.get('stat', {}).get('view')
                item['like_count'] = video_data.get('stat', {}).get('like')
                item['coin_count'] = video_data.get('stat', {}).get('coin')
                item['favorite_count'] = video_data.get('stat', {}).get('favorite')
                item['share_count'] = video_data.get('stat', {}).get('share')
                item['comment_count'] = video_data.get('stat', {}).get('reply')
                item['tags'] = [tag.get('tag_name') for tag in video_data.get('rcmd_reason', {}).get('tags', []) if tag]
                item['category'] = video_data.get('tname')
                item['top_comments'] = []  # 暂时不爬取评论
                
                # 使用yield将提取到的item交给Scrapy引擎处理（例如，保存到文件或数据库）
                self.crawled_count += 1
                yield item

            # --- 翻页逻辑 ---
            if self.crawled_count < self.target_count and page_number < self.max_pages and len(data['data']['list']) >= self.page_size:
                next_page_number = page_number + 1
                logging.info(f"正在准备请求下一页: {next_page_number}")
                next_page_url = f"https://api.bilibili.com/x/web-interface/popular?ps=20&pn={next_page_number}"
                yield scrapy.Request(
                    url=next_page_url,
                    callback=self.parse,
                    meta={'page_number': next_page_number},
                    headers=response.request.headers  # 将请求头传递给下一个请求
                )
        else:
            # 如果API返回错误，记录错误日志
            logging.error(f"获取第 {page_number} 页数据失败。 状态码: {data.get('code')}, 消息: {data.get('message')}")
            logging.error(f"响应体内容: {response.body.decode('utf-8')}")
