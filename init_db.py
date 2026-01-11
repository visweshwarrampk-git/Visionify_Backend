# init_db.py

"""
Database initialization script
Run this once to create all database tables
"""

from database import init_db, test_connection, engine, SessionLocal
from models import Base, User
from auth import hash_password
import os


def create_directories():
    """Create necessary directories"""
    images_dir = os.getenv("VIOLATION_IMAGES_DIR", "violation_images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"✅ Created directory: {images_dir}")


def create_default_super_admin():
    """Create default super admin if not exists"""
    db = SessionLocal()
    try:
        # Check if any super admin exists
        existing = db.query(User).filter(User.role == "super_admin").first()
        
        if not existing:
            print("\n👤 Creating default super admin...")
            default_admin = User(
                full_name="System Administrator",
                email="admin@safetyvision.com",
                phone="+1 (555) 000-0000",
                department="IT",
                role="super_admin",
                zone_access=["Zone A", "Zone B", "Zone C", "Zone D"],
                password_hash=hash_password("admin123"),
                is_active=True
            )
            db.add(default_admin)
            db.commit()
            print("✅ Default super admin created!")
            print("\n" + "=" * 60)
            print("📧 DEFAULT LOGIN CREDENTIALS")
            print("=" * 60)
            print("Email:    admin@safetyvision.com")
            print("Password: admin123")
            print("=" * 60)
            print("⚠️  IMPORTANT: Change this password after first login!")
            print("=" * 60)
        else:
            print("\n✅ Super admin already exists")
    
    except Exception as e:
        print(f"❌ Error creating default super admin: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


def main():
    print("=" * 60)
    print("🗄️  VISIONIFY DATABASE INITIALIZATION")
    print("=" * 60)
    
    # Test connection
    print("\n1️⃣  Testing database connection...")
    if not test_connection():
        print("\n❌ Cannot connect to database. Please check:")
        print("   - PostgreSQL is running")
        print("   - Database credentials in .env are correct")
        print("   - Database 'visionify_db' exists")
        print("\n📝 To create database, run:")
        print("   psql -U postgres")
        print("   CREATE DATABASE visionify_db;")
        print("   \\q")
        return
    
    # Create directories
    print("\n2️⃣  Creating storage directories...")
    create_directories()
    
    # Create tables
    print("\n3️⃣  Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        
        # List created tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\n📋 Created tables: {', '.join(tables)}")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create default super admin
    print("\n4️⃣  Setting up default user...")
    create_default_super_admin()
    
    print("\n" + "=" * 60)
    print("✅ DATABASE INITIALIZATION COMPLETE!")
    print("=" * 60)
    print("\nYou can now start the API server with:")
    print("  uvicorn api:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()