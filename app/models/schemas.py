from pydantic import BaseModel, Field
from typing import Optional, List, Any

class TopicCluster(BaseModel):
    cluster_id: int
    count: int
    avg_view: float
    top_keywords: List[str]
    representative_titles: List[str]

class AnalysisSummary(BaseModel):
    total_videos: int
    kmeans_clusters: dict
    dbscan_noise_ratio: float
    dbscan_clusters_count: int

class PredictRequest(BaseModel):
    title: str = Field(..., example="新手入门：3天做出爆款视频？")
    category: str = Field(default="未知", example="知识") # 设置默认值，使其变为可选
    author_id: Optional[int] = Field(None, example=123456)
    publish_time: Optional[int] = Field(None, example=1700000000)

class PredictViewResponse(BaseModel):
    predicted_view: float
    predicted_bucket: str
    bucket_id: int

class PredictBucketResponse(BaseModel):
    predicted_bucket: str
    bucket_id: int
    probabilities: Optional[dict] = None

class ErrorResponse(BaseModel):
    error: str
