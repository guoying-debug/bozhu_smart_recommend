import pandas as pd
import os
from sqlalchemy import create_engine
from app.core.config import DATA_PATH, get_database_url

_DF = None

def load_data():
    global _DF
    if _DF is not None:
        return _DF
    
    print("正在加载预处理数据...")
    if os.path.exists(DATA_PATH):
        _DF = pd.read_csv(DATA_PATH)
        print(f"成功从 {DATA_PATH} 加载了 {len(_DF)} 条记录。")
    else:
        # Fallback or raise error? For now return empty DF or None
        # In a real app, we might want to trigger a data processing pipeline
        print(f"警告: 找不到数据文件: {DATA_PATH}。")
        return None

    database_url = get_database_url()
    if database_url and "category" not in _DF.columns:
        try:
            engine = create_engine(database_url)
            # Optimize: load only necessary columns
            cat_df = pd.read_sql("SELECT video_id, category FROM videos", engine)
            _DF = _DF.merge(cat_df, on="video_id", how="left")
            _DF["category"] = _DF["category"].fillna("未知").astype(str)
        except Exception as e:
            print(f"数据库加载分类失败: {e}")
            pass
            
    return _DF

def get_data():
    return load_data()
