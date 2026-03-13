from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
import logging
import os
import json

class BilibiliDataValidationPipeline:
    """数据校验管道：过滤脏数据"""
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        # 1. 校验关键字段是否存在且非空
        required_fields = ['video_id', 'title', 'view_count']
        for field in required_fields:
            if not adapter.get(field):
                raise DropItem(f"Missing required field: {field} in {item}")

        # 2. 校验数值类型
        numeric_fields = ['view_count', 'like_count', 'coin_count', 'favorite_count', 'share_count']
        for field in numeric_fields:
            value = adapter.get(field)
            if value is None:
                adapter[field] = 0 # 缺失补0
                continue
                
            if not isinstance(value, (int, float)):
                try:
                    adapter[field] = int(value)
                except ValueError:
                    logging.warning(f"Invalid numeric value for {field}: {value}, setting to 0")
                    adapter[field] = 0
                    
        # 3. 校验 video_id 是否已存在 (简单增量去重逻辑)
        # 注意：这只是单次运行内的去重，或者基于已有文件的去重
        # 如果需要跨次运行增量，需要在 Spider 中加载已有 ID，或者在这里读取数据库/文件检查
        if hasattr(spider, 'seen_ids') and adapter['video_id'] in spider.seen_ids:
             raise DropItem(f"Duplicate video_id found: {adapter['video_id']}")
        
        if hasattr(spider, 'seen_ids'):
            spider.seen_ids.add(adapter['video_id'])

        return item

class BilibiliJsonWriterPipeline:
    """JSON 增量写入管道 (追加模式)"""
    def open_spider(self, spider):
        self.file = open('output_incremental.json', 'a', encoding='utf-8')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict(), ensure_ascii=False) + "\n"
        self.file.write(line)
        return item
