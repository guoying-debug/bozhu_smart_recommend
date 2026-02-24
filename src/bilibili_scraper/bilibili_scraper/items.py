import scrapy


class BilibiliScraperItem(scrapy.Item):
    video_id = scrapy.Field()
    title = scrapy.Field()
    description = scrapy.Field()
    author = scrapy.Field()
    author_id = scrapy.Field()
    publish_time = scrapy.Field()
    view_count = scrapy.Field()
    like_count = scrapy.Field()
    coin_count = scrapy.Field()
    favorite_count = scrapy.Field()
    share_count = scrapy.Field()
    comment_count = scrapy.Field()
    tags = scrapy.Field()
    category = scrapy.Field()
    top_comments = scrapy.Field()
