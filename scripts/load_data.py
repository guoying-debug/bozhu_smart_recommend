import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_setup import DATABASE_URL, Video # Reuse the Video model and DB URL
import json

# Define the path to the data file, relative to the project root.
# We need to go up one level from 'scripts' to the project root, then to 'src'.
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'bilibili_scraper', 'output.json')

def clean_data(df):
    """Applies the same cleaning steps as in the notebook."""
    # Convert Unix timestamp to datetime objects
    df['publish_time'] = pd.to_datetime(df['publish_time'], unit='s')

    # Ensure numeric columns are of the correct type
    numeric_cols = ['view_count', 'like_count', 'coin_count', 'favorite_count', 'share_count', 'comment_count']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Handle missing values
    df['description'] = df['description'].fillna('')
    
    # The 'tags' column is already a list of strings from the scraper,
    # but when read by pandas from JSON lines, it might be a string representation.
    # We ensure it's a proper list of strings for JSON conversion.
    # For simplicity, we'll handle this during insertion.
    
    return df

def load_data_to_db():
    """Loads data from the JSON file, cleans it, and inserts it into the database."""
    
    print(f"Reading data from {DATA_FILE_PATH}...")
    try:
        df = pd.read_json(DATA_FILE_PATH, lines=True, encoding='utf-8')
        print(f"Successfully loaded {len(df)} records from the file.")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_FILE_PATH}")
        return
    except Exception as e:
        print(f"ERROR: Failed to read or parse data file: {e}")
        return

    print("Cleaning data...")
    df_cleaned = clean_data(df.copy())
    
    print("Connecting to the database...")
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    print("Inserting data into the 'videos' table...")
    records_to_add = []
    for _, row in df_cleaned.iterrows():
        # Check if the record already exists
        exists = session.query(Video).filter_by(video_id=row['video_id']).first()
        if not exists:
            video_record = Video(
                video_id=row['video_id'],
                title=row['title'],
                description=row['description'],
                author=row['author'],
                author_id=row['author_id'],
                publish_time=row['publish_time'],
                view_count=row['view_count'],
                like_count=row['like_count'],
                coin_count=row['coin_count'],
                favorite_count=row['favorite_count'],
                share_count=row['share_count'],
                comment_count=row['comment_count'],
                tags=row['tags'], # SQLAlchemy handles Python list to JSON conversion
                category=row['category']
            )
            records_to_add.append(video_record)
        else:
            print(f"Skipping existing record: {row['video_id']}")

    if records_to_add:
        try:
            session.add_all(records_to_add)
            session.commit()
            print(f"Successfully inserted {len(records_to_add)} new records into the database.")
        except Exception as e:
            print(f"ERROR: Failed to insert records: {e}")
            session.rollback()
    else:
        print("No new records to insert.")
        
    session.close()
    print("Database session closed.")

if __name__ == "__main__":
    load_data_to_db()
