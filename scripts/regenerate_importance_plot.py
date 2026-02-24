
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance(df, output_path):
    """
    使用随机森林分析影响话题聚类的特征重要性, 并将图表保存到指定路径。
    """
    print("\n--- 开始进行特征重要性分析 ---")
    
    features = ['view_count', 'like_count', 'coin_count', 'favorite_count', 'share_count']
    target = 'bert_kmeans_cluster'
    
    # 确保目标列存在且不是全为NaN
    if target not in df.columns or df[target].isnull().all():
        print(f"错误: 目标列 '{target}' 不存在或全部为空。")
        return
        
    # 移除目标列为NaN的行
    df_cleaned = df.dropna(subset=[target])

    X = df_cleaned[features]
    y = df_cleaned[target]
    
    print(f"分析特征: {features}")
    print(f"目标变量: {target}")
    
    print("正在训练随机森林分类器...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print("模型训练完成。")
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性排名:")
    print(importance_df)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance for Topic Clustering', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    plt.savefig(output_path)
    plt.close()
    print(f"\n特征重要性图表已保存到: {output_path}")
    print("-" * 50)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'models', 'videos_with_clusters_and_coords.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'app', 'static', 'feature_importance.png')
    
    if not os.path.exists(DATA_PATH):
        print(f"错误: 数据文件未找到 at {DATA_PATH}")
    else:
        print(f"正在从 {DATA_PATH} 加载数据...")
        df = pd.read_csv(DATA_PATH)
        analyze_feature_importance(df, OUTPUT_PATH)
