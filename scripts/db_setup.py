import os
import getpass
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Configuration ---
# TODO: Move these settings to a separate config file or environment variables
DB_USER = os.getenv("DB_USER", "root")
# Securely get the password from an environment variable
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306") # Default MySQL port
DB_NAME = os.getenv("DB_NAME", "bilibili_data")

if not DB_PASSWORD:
    print("FATAL: Database password not found.")
    print("Please set the DB_PASSWORD environment variable before running the script.")
    print('Example: $env:DB_PASSWORD = "your_password"')
    exit(1)

# Construct the database URL for MySQL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Model Definition ---
Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'

    video_id = Column(String(32), primary_key=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    author = Column(String(255))
    author_id = Column(BigInteger)
    publish_time = Column(DateTime, nullable=False)
    view_count = Column(Integer)
    like_count = Column(Integer)
    coin_count = Column(Integer)
    favorite_count = Column(Integer)
    share_count = Column(Integer)
    comment_count = Column(Integer)
    tags = Column(JSON)
    category = Column(String(100))

    def __repr__(self):
        return f"<Video(title='{self.title[:30]}...')>"

# --- Database Setup Function ---
def setup_database():
    """
    Connects to the database, creates the 'videos' table if it doesn't exist.
    """
    try:
        # The engine is the starting point for any SQLAlchemy application.
        # 'echo=True' will log all generated SQL.
        engine = create_engine(DATABASE_URL, echo=True)
        
        print("Connecting to the database...")
        # Try to connect to the database
        with engine.connect() as connection:
            print("Database connection successful.")
        
        print("Creating table 'videos' if it does not exist...")
        # Create all tables in the engine. This is equivalent to "Create Table"
        # statements in raw SQL.
        Base.metadata.create_all(engine)
        print("Table 'videos' created successfully (if it didn't exist).")

    except Exception as e:
        print(f"An error occurred during database setup: {e}")
        print("\nPlease check the following:")
        print("1. Your MySQL server is running.")
        print("2. The database 'bilibili_data' exists (`CREATE DATABASE bilibili_data;`)")
        print("3. The host, port, and user in 'scripts/db_setup.py' are correct.")
        print("4. The password you entered is correct.")

if __name__ == "__main__":
    # This block will be executed when the script is run directly
    setup_database()
