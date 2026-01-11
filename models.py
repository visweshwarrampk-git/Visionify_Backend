# models.py

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Enum
from database import Base
import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from database import Base


class SeverityEnum(str, enum.Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class StatusEnum(str, enum.Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class Violation(Base):
    __tablename__ = "violations"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Violation Details
    violation_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    
    # Location & Source
    stream_id = Column(String(100), nullable=False, index=True)
    zone = Column(String(100), nullable=False, index=True)
    
    # Timestamp Information
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    day_of_week = Column(String(20), nullable=False)
    
    # Image Storage
    photo_path = Column(Text, nullable=True)
    
    # Description
    description = Column(Text, nullable=True)
    title = Column(String(200), nullable=True)
    
    # Status
    status = Column(String(20), nullable=False, default="new")
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Violation(id={self.id}, type={self.violation_type}, zone={self.zone}, timestamp={self.timestamp})>"

    def to_dict(self):
        """Convert model to dictionary - REQUIRED METHOD"""
        return {
            "id": self.id,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "type": self.severity,  # ✅ Frontend compatibility
            "violation": self.violation_type,  # ✅ Frontend compatibility
            "stream_id": self.stream_id,
            "zone": self.zone,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "day_of_week": self.day_of_week,
            "photo_path": self.photo_path,
            "description": self.description,
            "title": self.title,
            "status": self.status,
            "source": "AI Detection",  # ✅ Frontend compatibility
            "read": False,  # ✅ Frontend compatibility
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
# ✅ NEW USER MODEL
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    full_name = Column(String(200), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20), nullable=True)
    department = Column(String(100), nullable=True)
    role = Column(String(50), nullable=False, index=True)  # super_admin or admin
    zone_access = Column(JSON, nullable=True)  # ["Zone A", "Zone B"]
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary (never include password)"""
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "department": self.department,
            "role": self.role,
            "zone_access": self.zone_access or [],
            "is_active": self.is_active,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }