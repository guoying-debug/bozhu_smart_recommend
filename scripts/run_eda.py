import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
import numpy as np # Import numpy for log transformation

# Set plot style and handle potential font issues for non-English characters
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use a font that supports Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative signs

# --- Database Configuration ---
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD") # Assumes you've set this environment variable
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "bilibili_data")

# Construct the database URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def analyze_and_visualize(df):
    """
    Performs descriptive statistics, creates visualizations, and applies feature engineering.
    """
    print("\n3. Descriptive Statistics for Numeric Columns:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numeric_cols].describe())
    print("-" * 50)

    print("\n4. Generating and saving distribution plots for raw data...")
    metrics_to_plot = ['view_count', 'like_count', 'comment_count', 'favorite_count']
    for col in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Raw {col} Distribution (原始数据分布)')
        plt.xlabel(col)
        plt.ylabel('Frequency (频率)')
        plot_path = os.path.join('plots', f'raw_{col}_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"   - Saved {plot_path}")
    print("-" * 50)

    print("\n5. Feature Engineering: Log Transformation...")
    # We use log1p which calculates log(1 + x) to handle zero values
    for col in metrics_to_plot:
        df[f'log_{col}'] = np.log1p(df[col])
    
    print("   - Created log-transformed features.")
    print("-" * 50)

    print("\n6. Generating and saving distribution plots for log-transformed data...")
    for col in metrics_to_plot:
        log_col_name = f'log_{col}'
        plt.figure(figsize=(10, 6))
        sns.histplot(df[log_col_name], kde=True, bins=20)
        plt.title(f'Log-Transformed {col} Distribution (对数变换后分布)')
        plt.xlabel(log_col_name)
        plt.ylabel('Frequency (频率)')
        plot_path = os.path.join('plots', f'log_{col}_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"   - Saved {plot_path}")
    print("-" * 50)

    print("\n7. Generating and saving correlation heatmap...")
    # Select original numeric columns for correlation analysis
    corr_matrix = df[metrics_to_plot].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Key Metrics (关键指标相关性热力图)')
    plot_path = os.path.join('plots', 'correlation_heatmap.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   - Saved {plot_path}")
    print("\nAll plots have been saved to the 'plots' directory.")
    print("-" * 50)


def perform_eda():
    """
    Connects to the database, loads video data, and performs full EDA.
    """
    if not DB_PASSWORD:
        print("FATAL: DB_PASSWORD environment variable is not set.")
        print("Please set it before running the script, e.g., in PowerShell:")
        print('$env:DB_PASSWORD = "your_password"')
        return

    try:
        print("Connecting to the database...")
        engine = create_engine(DATABASE_URL)
        
        query = "SELECT * FROM videos;"
        
        print("Loading data into DataFrame...")
        df = pd.read_sql(query, engine)
        
        print("Successfully loaded data from the database.")
        print("-" * 50)
        
        print("\n1. First 5 rows of the dataset (df.head()):")
        print(df.head())
        print("-" * 50)
        
        print("\n2. DataFrame Info (df.info()):")
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        print(s)
        print("-" * 50)

        analyze_and_visualize(df)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    perform_eda()
