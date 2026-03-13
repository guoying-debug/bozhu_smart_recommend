import scrapy
import json
import logging
import math
import os
from bilibili_scraper.items import BilibiliScraperItem

class VideoSpider(scrapy.Spider):
    name = "video"
    allowed_domains = ["api.bilibili.com"]
    
    def __init__(self, target_count=5000, incremental=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_size = 20
        self.target_count = int(target_count)
        self.max_pages = max(1, int(math.ceil(self.target_count / self.page_size)))
        self.crawled_count = 0
        self.incremental = str(incremental).lower() == 'true'
        self.seen_ids = set()
        
        # 如果是增量抓取，先加载已有的 ID
        if self.incremental:
            self._load_seen_ids()

    def _load_seen_ids(self):
        """加载已存在的视频 ID，防止重复抓取"""
        # 这里假设从 output.json 加载，实际生产环境应该查数据库或 Redis
        # 为了演示，我们尝试读取 raw data 目录下的文件
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'data', 'raw', 'output.json')
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if 'video_id' in item:
                            self.seen_ids.add(item['video_id'])
                logging.info(f"Loaded {len(self.seen_ids)} existing video IDs for incremental crawling.")
            except Exception as e:
                logging.warning(f"Failed to load existing data for incremental crawling: {e}")

    def start_requests(self):
        headers = {
            'Referer': 'https://www.bilibili.com/v/popular/rank/all',
        }
        # 热门接口
        yield scrapy.Request(
            url="https://api.bilibili.com/x/web-interface/popular?ps=20&pn=1",
            callback=self.parse,
            meta={'page_number': 1},
            headers=headers
        )

    def parse(self, response):
        page_number = response.meta['page_number']
        logging.info(f"--- Parsing Page {page_number} ---")
        
        try:
            data = json.loads(response.body)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from {response.url}")
            return

        if data.get('code') == 0 and data.get('data', {}).get('list'):
            video_list = data['data']['list']
            logging.info(f"Got {len(video_list)} items on page {page_number}.")
            
            new_items_count = 0
            
            for video_data in video_list:
                if self.crawled_count >= self.target_count:
                    break
                
                bvid = video_data.get('bvid')
                
                # 增量抓取检查
                if self.incremental and bvid in self.seen_ids:
                    logging.debug(f"Skipping existing video: {bvid}")
                    continue

                item = BilibiliScraperItem()
                item['video_id'] = bvid
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
                
                # 安全获取 tags
                rcmd_reason = video_data.get('rcmd_reason')
                if isinstance(rcmd_reason, dict):
                     item['tags'] = [tag.get('tag_name') for tag in rcmd_reason.get('tags', []) if tag]
                else:
                    item['tags'] = []

                item['category'] = video_data.get('tname')
                item['top_comments'] = [] 
                
                self.crawled_count += 1
                new_items_count += 1
                yield item

            # 翻页逻辑
            # 如果是增量抓取，且当前页所有数据都是旧的，可能意味着后面也都是旧的（对于时间排序的列表），可以考虑提前停止
            # 但热门列表不完全按时间，所以还是建议继续爬
            if self.crawled_count < self.target_count and page_number < self.max_pages and len(video_list) >= self.page_size:
                next_page_number = page_number + 1
                next_page_url = f"https://api.bilibili.com/x/web-interface/popular?ps=20&pn={next_page_number}"
                yield scrapy.Request(
                    url=next_page_url,
                    callback=self.parse,
                    meta={'page_number': next_page_number},
                    headers=response.request.headers
                )
        else:
            logging.error(f"Failed to fetch page {page_number}. Code: {data.get('code')}, Message: {data.get('message')}")
