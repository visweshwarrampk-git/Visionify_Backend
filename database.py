# database.py

import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/visionify_db"
)

# Create database engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database - create all tables"""
    from models import Violation  # Import after Base is defined
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def test_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        # Use text() for raw SQL in SQLAlchemy 2.0+
        db.execute(text("SELECT 1"))
        db.close()
        print("✅ Database connection successful!")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print(f"\n🔍 Troubleshooting:")
        print(f"   1. Check if PostgreSQL is running")
        print(f"   2. Verify password in .env file")
        print(f"   3. Ensure database 'visionify_db' exists")
        print(f"   4. Check DATABASE_URL in .env")
        return False