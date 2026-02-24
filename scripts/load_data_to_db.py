import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import os
import json

# Assuming db_setup.py is in the same directory or accessible
from db_setup import Base, Video, DATABASE_URL

def clean_data(df):
    """
    Cleans and preprocesses the DataFrame.
    """
    print("Starting data cleaning and preprocessing...")
    
    # 1. Convert Unix timestamp to datetime
    df['publish_time'] = pd.to_datetime(df['publish_time'], unit='s')

    # 2. Ensure numeric columns are correct type and handle missing values
    numeric_cols = ['view_count', 'like_count', 'coin_count', 'favorite_count', 'share_count', 'comment_count', 'author_id']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 3. Handle missing text values
    df['description'] = df['description'].fillna('')
    df['author'] = df['author'].fillna('Unknown')
    df['category'] = df['category'].fillna('Uncategorized')

    # 4. Handle 'tags' which should be a list (JSON)
    # The 'tags' from scrapy should already be a list. If not, this helps.
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

    # 5. Drop duplicates based on video_id to be safe
    df.drop_duplicates(subset=['video_id'], keep='first', inplace=True)
    
    print("Data cleaning finished.")
    return df

def load_data_to_db():
    """
    Loads data from JSON file, cleans it, and upserts into the database.
    """
    file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'bilibili_scraper', 'output.json')

    try:
        # Load data from JSON file
        print(f"Loading data from {file_path}...")
        # The output is a JSON array, so we set lines=False (or omit it)
        df = pd.read_json(file_path, lines=False, encoding='utf-8')
        print(f"Loaded {len(df)} records from file.")

        # Clean the data
        df = clean_data(df)

        # --- Database Interaction ---
        if not DB_PASSWORD:
            print("FATAL: DB_PASSWORD environment variable is not set.")
            return
            
        try:
            engine = create_engine(DATABASE_URL)
            Session = sessionmaker(bind=engine)
            session = Session()

            # Step 1: Truncate the table to remove old data
            print("Truncating 'videos' table to clear old data...")
            session.execute(text("TRUNCATE TABLE videos;"))
            print("Table 'videos' truncated.")

            # Step 2: Insert new data
            print(f"Inserting {len(df)} new records into the database...")
            records = df.to_dict(orient='records')
            
            session.bulk_insert_mappings(Video, records)
            
            session.commit()
            print("Successfully inserted new data into the database.")

        except IntegrityError as e:
            print(f"Integrity Error: {e}. This might be due to duplicate keys if not handled.")
            session.rollback()
        except Exception as e:
            print(f"An error occurred during database operation: {e}")
            if 'session' in locals():
                session.rollback()
        finally:
            if 'session' in locals():
                session.close()
    
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    load_data_to_db()
