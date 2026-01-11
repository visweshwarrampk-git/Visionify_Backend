# crud.py

import os
import cv2
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from models import Violation
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from sqlalchemy import func, case
from models import User
from auth import hash_password, verify_password
from sqlalchemy import func, extract


def save_violation_image(frame: np.ndarray, violation_type: str, stream_id: str) -> Optional[str]:
    """
    Save violation image to disk
    
    Args:
        frame: OpenCV image (numpy array)
        violation_type: Type of violation
        stream_id: Stream identifier
    
    Returns:
        Path to saved image or None if failed
    """
    try:
        # Create directory if not exists
        images_dir = os.getenv("VIOLATION_IMAGES_DIR", "violation_images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{violation_type}_{stream_id}_{timestamp}.jpg"
        filepath = os.path.join(images_dir, filename)
        
        # Save image
        success = cv2.imwrite(filepath, frame)
        
        if success:
            print(f"✅ Image saved: {filepath}")
            return filepath
        else:
            print(f"❌ Failed to save image: {filepath}")
            return None
            
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return None


def create_violation(
    db: Session,
    violation_type: str,
    stream_id: str,
    zone: str,
    severity: str,
    photo_path: Optional[str] = None,
    description: Optional[str] = None,
    title: Optional[str] = None
) -> Violation:
    """
    Create a new violation record
    
    Args:
        db: Database session
        violation_type: Type of violation
        stream_id: Stream identifier
        zone: Zone name
        severity: Severity level (critical, warning, info)
        photo_path: Path to violation image
        description: Violation description
        title: Violation title
    
    Returns:
        Created Violation object
    """
    try:
        # Get current timestamp and day
        now = datetime.now()
        day_of_week = now.strftime("%A")  # Monday, Tuesday, etc.
        
        # Create violation object
        violation = Violation(
            violation_type=violation_type,
            severity=severity,
            stream_id=stream_id,
            zone=zone,
            timestamp=now,
            day_of_week=day_of_week,
            photo_path=photo_path,
            description=description,
            title=title,
            status="new"
        )
        
        # Add to database
        db.add(violation)
        db.commit()
        db.refresh(violation)
        
        print(f"✅ Violation saved to DB: ID={violation.id}, Type={violation_type}, Zone={zone}")
        return violation
        
    except Exception as e:
        print(f"❌ Error creating violation: {e}")
        db.rollback()
        raise


def get_violation(db: Session, violation_id: int) -> Optional[Violation]:
    """Get violation by ID"""
    return db.query(Violation).filter(Violation.id == violation_id).first()


def get_violations(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    violation_type: Optional[str] = None,
    zone: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Violation]:
    """
    Get violations with optional filters
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        violation_type: Filter by violation type
        zone: Filter by zone
        severity: Filter by severity
        status: Filter by status
        start_date: Filter by start date
        end_date: Filter by end date
    
    Returns:
        List of Violation objects
    """
    query = db.query(Violation)
    
    # Apply filters
    if violation_type:
        query = query.filter(Violation.violation_type == violation_type)
    
    if zone:
        query = query.filter(Violation.zone == zone)
    
    if severity:
        query = query.filter(Violation.severity == severity)
    
    if status:
        query = query.filter(Violation.status == status)
    
    if start_date:
        query = query.filter(Violation.timestamp >= start_date)
    
    if end_date:
        query = query.filter(Violation.timestamp <= end_date)
    
    # Order by most recent first
    query = query.order_by(desc(Violation.timestamp))
    
    # Apply pagination
    return query.offset(skip).limit(limit).all()


def update_violation_status(
    db: Session,
    violation_id: int,
    status: str
) -> Optional[Violation]:
    """Update violation status"""
    violation = get_violation(db, violation_id)
    
    if violation:
        violation.status = status
        violation.updated_at = datetime.now()
        db.commit()
        db.refresh(violation)
        print(f"✅ Violation {violation_id} status updated to: {status}")
        return violation
    
    return None


def delete_violation(db: Session, violation_id: int) -> bool:
    """Delete violation by ID"""
    violation = get_violation(db, violation_id)
    
    if violation:
        # Delete image file if exists
        if violation.photo_path and os.path.exists(violation.photo_path):
            try:
                os.remove(violation.photo_path)
                print(f"✅ Deleted image: {violation.photo_path}")
            except Exception as e:
                print(f"⚠️ Could not delete image: {e}")
        
        # Delete from database
        db.delete(violation)
        db.commit()
        print(f"✅ Violation {violation_id} deleted")
        return True
    
    return False


def get_violation_stats(db: Session) -> dict:
    """Get violation statistics"""
    total = db.query(func.count(Violation.id)).scalar()
    critical = db.query(func.count(Violation.id)).filter(Violation.severity == "critical").scalar()
    warnings = db.query(func.count(Violation.id)).filter(Violation.severity == "warning").scalar()
    new_count = db.query(func.count(Violation.id)).filter(Violation.status == "new").scalar()
    
    return {
        "total": total,
        "critical": critical,
        "warnings": warnings,
        "new": new_count
    }


def get_violations_by_zone(db: Session) -> dict:
    """Get violation count grouped by zone"""
    results = db.query(
        Violation.zone,
        func.count(Violation.id).label("count")
    ).group_by(Violation.zone).all()
    
    return {zone: count for zone, count in results}


def get_violations_by_type(db: Session) -> dict:
    """Get violation count grouped by type"""
    results = db.query(
        Violation.violation_type,
        func.count(Violation.id).label("count")
    ).group_by(Violation.violation_type).all()
    
    return {vtype: count for vtype, count in results}


# ======================================================
# REPORT GENERATION FUNCTIONS
# ======================================================

def get_daily_report(db: Session, date: str) -> Dict[str, Any]:
    """
    Generate daily report for a specific date
    
    Args:
        db: Database session
        date: Date string in format YYYY-MM-DD
    
    Returns:
        Dictionary with report data
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        next_date = target_date + timedelta(days=1)
        
        # Get all violations for the day
        violations = db.query(Violation).filter(
            and_(
                Violation.timestamp >= target_date,
                Violation.timestamp < next_date
            )
        ).order_by(desc(Violation.timestamp)).all()
        
        # Summary stats
        total = len(violations)
        critical = sum(1 for v in violations if v.severity == "critical")
        warnings = sum(1 for v in violations if v.severity == "warning")
        info = sum(1 for v in violations if v.severity == "info")
        
        # Count by type
        by_type = {}
        for v in violations:
            by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        
        # Count by zone
        by_zone = {}
        for v in violations:
            by_zone[v.zone] = by_zone.get(v.zone, 0) + 1
        
        # Count by hour
        by_hour = {}
        for v in violations:
            hour = v.timestamp.hour
            by_hour[hour] = by_hour.get(hour, 0) + 1
        
        return {
            "report_type": "daily",
            "date": date,
            "summary": {
                "total": total,
                "critical": critical,
                "warnings": warnings,
                "info": info
            },
            "breakdown": {
                "by_type": by_type,
                "by_zone": by_zone,
                "by_hour": by_hour
            },
            "violations": [v.to_dict() for v in violations]
        }
        
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD")


def get_weekly_report(db: Session, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Generate weekly report for a date range
    
    Args:
        db: Database session
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        Dictionary with report data
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date() + timedelta(days=1)
        
        # Get all violations in range
        violations = db.query(Violation).filter(
            and_(
                Violation.timestamp >= start,
                Violation.timestamp < end
            )
        ).order_by(desc(Violation.timestamp)).all()
        
        # Summary stats
        total = len(violations)
        critical = sum(1 for v in violations if v.severity == "critical")
        warnings = sum(1 for v in violations if v.severity == "warning")
        info = sum(1 for v in violations if v.severity == "info")
        
        # Count by type
        by_type = {}
        for v in violations:
            by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        
        # Count by zone
        by_zone = {}
        for v in violations:
            by_zone[v.zone] = by_zone.get(v.zone, 0) + 1
        
        # Count by day of week
        by_day = {}
        for v in violations:
            day = v.day_of_week
            by_day[day] = by_day.get(day, 0) + 1
        
        # Daily breakdown
        by_date = {}
        for v in violations:
            date_key = v.timestamp.strftime("%Y-%m-%d")
            by_date[date_key] = by_date.get(date_key, 0) + 1
        
        return {
            "report_type": "weekly",
            "start_date": start_date,
            "end_date": end_date,
            "summary": {
                "total": total,
                "critical": critical,
                "warnings": warnings,
                "info": info
            },
            "breakdown": {
                "by_type": by_type,
                "by_zone": by_zone,
                "by_day_of_week": by_day,
                "by_date": by_date
            },
            "violations": [v.to_dict() for v in violations]
        }
        
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD")


def get_monthly_report(db: Session, year: int, month: int) -> Dict[str, Any]:
    """
    Generate monthly report
    
    Args:
        db: Database session
        year: Year (e.g., 2026)
        month: Month (1-12)
    
    Returns:
        Dictionary with report data
    """
    try:
        # Calculate start and end dates
        start_date = datetime(year, month, 1)
        
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        # Get all violations in month
        violations = db.query(Violation).filter(
            and_(
                Violation.timestamp >= start_date,
                Violation.timestamp < end_date
            )
        ).order_by(desc(Violation.timestamp)).all()
        
        # Summary stats
        total = len(violations)
        critical = sum(1 for v in violations if v.severity == "critical")
        warnings = sum(1 for v in violations if v.severity == "warning")
        info = sum(1 for v in violations if v.severity == "info")
        
        # Count by type
        by_type = {}
        for v in violations:
            by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        
        # Count by zone
        by_zone = {}
        for v in violations:
            by_zone[v.zone] = by_zone.get(v.zone, 0) + 1
        
        # Count by day of week
        by_day = {}
        for v in violations:
            day = v.day_of_week
            by_day[day] = by_day.get(day, 0) + 1
        
        # Count by week
        by_week = {}
        for v in violations:
            week_num = v.timestamp.isocalendar()[1]
            week_key = f"Week {week_num}"
            by_week[week_key] = by_week.get(week_key, 0) + 1
        
        # Daily breakdown
        by_date = {}
        for v in violations:
            date_key = v.timestamp.strftime("%Y-%m-%d")
            by_date[date_key] = by_date.get(date_key, 0) + 1
        
        return {
            "report_type": "monthly",
            "year": year,
            "month": month,
            "month_name": start_date.strftime("%B %Y"),
            "summary": {
                "total": total,
                "critical": critical,
                "warnings": warnings,
                "info": info
            },
            "breakdown": {
                "by_type": by_type,
                "by_zone": by_zone,
                "by_day_of_week": by_day,
                "by_week": by_week,
                "by_date": by_date
            },
            "violations": [v.to_dict() for v in violations]
        }
        
    except ValueError:
        raise ValueError("Invalid year or month")


def export_violations_to_csv(violations: List[Violation]) -> str:
    """
    Convert violations to CSV format
    
    Args:
        violations: List of Violation objects
    
    Returns:
        CSV string
    """
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow([
        "ID",
        "Violation Type",
        "Severity",
        "Stream ID",
        "Zone",
        "Timestamp",
        "Day of Week",
        "Description",
        "Status",
        "Photo Path"
    ])
    
    # Data
    for v in violations:
        writer.writerow([
            v.id,
            v.violation_type,
            v.severity,
            v.stream_id,
            v.zone,
            v.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            v.day_of_week,
            v.description,
            v.status,
            v.photo_path
        ])
    
    return output.getvalue()


# ======================================================
# DASHBOARD ANALYTICS FUNCTIONS
# ======================================================

def get_compliance_trend(
    db: Session,
    period: str = "weekly",
    date: str = None
) -> Dict[str, Any]:
    """
    Get compliance trend data for dashboard
    
    Args:
        db: Database session
        period: 'daily', 'weekly', or 'monthly'
        date: Reference date (YYYY-MM-DD), defaults to today
    
    Returns:
        Dictionary with labels and compliance percentages
    """
    try:
        if date:
            ref_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            ref_date = datetime.now().date()
        
        if period == "daily":
            # Last 7 days
            labels = []
            data = []
            
            for i in range(6, -1, -1):
                day = ref_date - timedelta(days=i)
                day_name = day.strftime("%a")  # Mon, Tue, etc.
                
                # Count violations for this day
                day_start = datetime.combine(day, datetime.min.time())
                day_end = datetime.combine(day, datetime.max.time())
                
                violation_count = db.query(func.count(Violation.id)).filter(
                    and_(
                        Violation.timestamp >= day_start,
                        Violation.timestamp <= day_end
                    )
                ).scalar() or 0
                
                # Calculate compliance (assuming 100% - violations as percentage)
                # You can adjust this formula based on your business logic
                compliance = max(0, 100 - (violation_count * 2))
                
                labels.append(day_name)
                data.append(compliance)
        
        elif period == "weekly":
            # Last 7 weeks
            labels = []
            data = []
            
            for i in range(6, -1, -1):
                week_start = ref_date - timedelta(weeks=i, days=ref_date.weekday())
                week_end = week_start + timedelta(days=6)
                week_label = f"W{week_start.isocalendar()[1]}"
                
                week_start_dt = datetime.combine(week_start, datetime.min.time())
                week_end_dt = datetime.combine(week_end, datetime.max.time())
                
                violation_count = db.query(func.count(Violation.id)).filter(
                    and_(
                        Violation.timestamp >= week_start_dt,
                        Violation.timestamp <= week_end_dt
                    )
                ).scalar() or 0
                
                compliance = max(0, 100 - (violation_count * 1.5))
                
                labels.append(week_label)
                data.append(round(compliance, 1))
        
        elif period == "monthly":
            # Last 7 months
            labels = []
            data = []
            
            for i in range(6, -1, -1):
                month_date = ref_date.replace(day=1) - timedelta(days=i*30)
                month_start = month_date.replace(day=1)
                
                if month_start.month == 12:
                    month_end = month_start.replace(year=month_start.year + 1, month=1, day=1) - timedelta(days=1)
                else:
                    month_end = month_start.replace(month=month_start.month + 1, day=1) - timedelta(days=1)
                
                month_label = month_start.strftime("%b")
                
                month_start_dt = datetime.combine(month_start, datetime.min.time())
                month_end_dt = datetime.combine(month_end, datetime.max.time())
                
                violation_count = db.query(func.count(Violation.id)).filter(
                    and_(
                        Violation.timestamp >= month_start_dt,
                        Violation.timestamp <= month_end_dt
                    )
                ).scalar() or 0
                
                compliance = max(0, 100 - (violation_count * 1))
                
                labels.append(month_label)
                data.append(round(compliance, 1))
        
        else:
            raise ValueError("Invalid period. Use 'daily', 'weekly', or 'monthly'")
        
        return {
            "labels": labels,
            "data": data,
            "period": period
        }
    
    except Exception as e:
        print(f"❌ Error getting compliance trend: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_top_violations(
    db: Session,
    period: str = "weekly",
    date: str = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get top violations data for pie chart
    
    Args:
        db: Database session
        period: 'daily', 'weekly', or 'monthly'
        date: Reference date (YYYY-MM-DD), defaults to today
        limit: Number of top violations to return
    
    Returns:
        Dictionary with violation distribution
    """
    try:
        if date:
            ref_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            ref_date = datetime.now().date()
        
        # Calculate date range based on period
        if period == "daily":
            start_date = datetime.combine(ref_date, datetime.min.time())
            end_date = datetime.combine(ref_date, datetime.max.time())
        elif period == "weekly":
            start_date = datetime.combine(
                ref_date - timedelta(days=ref_date.weekday()),
                datetime.min.time()
            )
            end_date = datetime.combine(
                start_date.date() + timedelta(days=6),
                datetime.max.time()
            )
        elif period == "monthly":
            start_date = datetime.combine(
                ref_date.replace(day=1),
                datetime.min.time()
            )
            if start_date.month == 12:
                end_date = datetime.combine(
                    start_date.replace(year=start_date.year + 1, month=1, day=1).date() - timedelta(days=1),
                    datetime.max.time()
                )
            else:
                end_date = datetime.combine(
                    start_date.replace(month=start_date.month + 1, day=1).date() - timedelta(days=1),
                    datetime.max.time()
                )
        else:
            raise ValueError("Invalid period")
        
        # Query top violations
        results = db.query(
            Violation.violation_type,
            func.count(Violation.id).label("count")
        ).filter(
            and_(
                Violation.timestamp >= start_date,
                Violation.timestamp <= end_date
            )
        ).group_by(
            Violation.violation_type
        ).order_by(
            func.count(Violation.id).desc()
        ).limit(limit).all()
        
        # Color mapping for violations
        color_map = {
            "no_helmet": "#ff1455",
            "no_gloves": "#4ade80",
            "no_goggle": "#8b5cf6",
            "no_boots": "#06b6d4",
            "no_vest": "#ff8b2c",
            "mobile_phone": "#3b82f6",
            "fall": "#ef4444",
            "fire": "#dc2626",
            "smoke": "#6b7280",
            "restricted_area": "#fbbf24"
        }
        
        violations = []
        total = 0
        
        for violation_type, count in results:
            # Format violation name
            name = violation_type.replace("_", " ").title()
            if name == "No Helmet":
                display_name = "No Helmet"
            elif name == "No Gloves":
                display_name = "No Gloves"
            elif name == "No Vest":
                display_name = "No Vest"
            elif name == "No Goggle":
                display_name = "No Goggles"
            elif name == "No Boots":
                display_name = "No Boots"
            elif name == "Mobile Phone":
                display_name = "Mobile Phone"
            elif name == "Restricted Area":
                display_name = "Unsafe Zone"
            else:
                display_name = name
            
            violations.append({
                "name": display_name,
                "count": count,
                "color": color_map.get(violation_type, "#94a3b8")
            })
            total += count
        
        return {
            "violations": violations,
            "total": total,
            "period": period
        }
    
    except Exception as e:
        print(f"❌ Error getting top violations: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_recent_incidents(
    db: Session,
    period: str = "all",
    date: str = None,
    limit: int = 3,
    skip: int = 0
) -> Dict[str, Any]:
    """
    Get recent incidents for dashboard with pagination
    
    Args:
        db: Database session
        period: 'daily', 'weekly', 'monthly', or 'all'
        date: Reference date (YYYY-MM-DD), defaults to today
        limit: Number of incidents to return (default 3)
        skip: Number of incidents to skip for pagination (default 0)
    
    Returns:
        Dictionary with incidents list and total count
    """
    try:
        if date:
            ref_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            ref_date = datetime.now().date()
        
        # Build query based on period
        query = db.query(Violation)
        
        if period == "daily":
            start_date = datetime.combine(ref_date, datetime.min.time())
            end_date = datetime.combine(ref_date, datetime.max.time())
            query = query.filter(
                and_(
                    Violation.timestamp >= start_date,
                    Violation.timestamp <= end_date
                )
            )
        elif period == "weekly":
            week_start = ref_date - timedelta(days=ref_date.weekday())
            week_end = week_start + timedelta(days=6)
            start_date = datetime.combine(week_start, datetime.min.time())
            end_date = datetime.combine(week_end, datetime.max.time())
            query = query.filter(
                and_(
                    Violation.timestamp >= start_date,
                    Violation.timestamp <= end_date
                )
            )
        elif period == "monthly":
            month_start = ref_date.replace(day=1)
            if month_start.month == 12:
                month_end = month_start.replace(year=month_start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = month_start.replace(month=month_start.month + 1, day=1) - timedelta(days=1)
            start_date = datetime.combine(month_start, datetime.min.time())
            end_date = datetime.combine(month_end, datetime.max.time())
            query = query.filter(
                and_(
                    Violation.timestamp >= start_date,
                    Violation.timestamp <= end_date
                )
            )
        # If period == "all", no date filter
        
        # Get total count before pagination
        total_count = query.count()
        
        # Order by most recent and apply pagination
        violations = query.order_by(desc(Violation.timestamp)).offset(skip).limit(limit).all()
        
        incidents = []
        for v in violations:
            # Calculate time ago
            now = datetime.now()
            diff = now - v.timestamp
            
            if diff.days > 0:
                time_ago = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            else:
                time_ago = "Just now"
            
            incidents.append({
                "id": v.id,
                "severity": v.severity,
                "zone": v.zone,
                "time_ago": time_ago,
                "title": v.title or f"{v.violation_type.replace('_', ' ').title()} Detected",
                "description": v.description or "",
                "timestamp": v.timestamp.isoformat()
            })
        
        return {
            "incidents": incidents,
            "total": total_count,
            "limit": limit,
            "skip": skip
        }
    
    except Exception as e:
        print(f"❌ Error getting recent incidents: {e}")
        import traceback
        traceback.print_exc()
        return {
            "incidents": [],
            "total": 0,
            "limit": limit,
            "skip": skip
        }


def get_zone_risk_status(
    db: Session,
    period: str = "weekly",
    date: str = None
) -> List[Dict[str, Any]]:
    """
    Get zone risk status for dashboard
    
    Args:
        db: Database session
        period: 'daily', 'weekly', or 'monthly'
        date: Reference date (YYYY-MM-DD), defaults to today
    
    Returns:
        List of zone risk data
    """
    try:
        if date:
            ref_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            ref_date = datetime.now().date()
        
        # Calculate date range
        if period == "daily":
            start_date = datetime.combine(ref_date, datetime.min.time())
            end_date = datetime.combine(ref_date, datetime.max.time())
        elif period == "weekly":
            week_start = ref_date - timedelta(days=ref_date.weekday())
            week_end = week_start + timedelta(days=6)
            start_date = datetime.combine(week_start, datetime.min.time())
            end_date = datetime.combine(week_end, datetime.max.time())
        elif period == "monthly":
            month_start = ref_date.replace(day=1)
            if month_start.month == 12:
                month_end = month_start.replace(year=month_start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = month_start.replace(month=month_start.month + 1, day=1) - timedelta(days=1)
            start_date = datetime.combine(month_start, datetime.min.time())
            end_date = datetime.combine(month_end, datetime.max.time())
        else:
            raise ValueError("Invalid period")
        
        # Get all zones with violations
        zone_data = db.query(
            Violation.zone,
            func.count(Violation.id).label("incident_count")
        ).filter(
            and_(
                Violation.timestamp >= start_date,
                Violation.timestamp <= end_date
            )
        ).group_by(Violation.zone).all()
        
        zones = []
        for zone_name, incident_count in zone_data:
            # Calculate compliance (simple formula: 100 - incidents * factor)
            # Adjust the factor based on your business logic
            compliance = max(0, min(100, 100 - (incident_count * 2)))
            
            # Determine risk level
            if compliance >= 90:
                risk_level = "low"
                risk_color = "bg-emerald-500"
                risk_dot = "bg-emerald-500"
            elif compliance >= 80:
                risk_level = "medium"
                risk_color = "bg-orange-400"
                risk_dot = "bg-orange-400"
            else:
                risk_level = "high"
                risk_color = "bg-red-500"
                risk_dot = "bg-red-500"
            
            # Calculate progress bar width
            width_percent = min(100, max(0, compliance))
            
            zones.append({
                "zone": zone_name,
                "incident_count": incident_count,
                "compliance": round(compliance, 0),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "risk_dot": risk_dot,
                "width_percent": width_percent
            })
        
        # Sort by incident count (highest risk first)
        zones.sort(key=lambda x: x["incident_count"], reverse=True)
        
        return zones
    
    except Exception as e:
        print(f"❌ Error getting zone risk status: {e}")
        import traceback
        traceback.print_exc()
        return []


# ======================================================
# USER MANAGEMENT FUNCTIONS
# ======================================================

def create_user(
    db: Session,
    full_name: str,
    email: str,
    password: str,
    role: str,
    phone: Optional[str] = None,
    department: Optional[str] = None,
    zone_access: Optional[List[str]] = None
) -> User:
    """Create a new user"""
    try:
        # Check if email already exists
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise ValueError(f"User with email {email} already exists")
        
        # Validate role
        if role not in ["super_admin", "admin"]:
            raise ValueError("Role must be 'super_admin' or 'admin'")
        
        # Create user
        user = User(
            full_name=full_name,
            email=email,
            password_hash=hash_password(password),
            role=role,
            phone=phone,
            department=department,
            zone_access=zone_access or [],
            is_active=True
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        print(f"✅ User created: {email} ({role})")
        return user
    
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating user: {e}")
        raise


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def get_all_users(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    role: Optional[str] = None,
    is_active: Optional[bool] = None
) -> List[User]:
    """Get all users with filters"""
    query = db.query(User)
    
    if role:
        query = query.filter(User.role == role)
    
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    
    return query.offset(skip).limit(limit).all()


def update_user(
    db: Session,
    user_id: int,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    department: Optional[str] = None,
    role: Optional[str] = None,
    zone_access: Optional[List[str]] = None,
    is_active: Optional[bool] = None,
    password: Optional[str] = None
) -> Optional[User]:
    """Update user details"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            return None
        
        if full_name is not None:
            user.full_name = full_name
        if email is not None:
            existing = db.query(User).filter(
                User.email == email, 
                User.id != user_id
            ).first()
            if existing:
                raise ValueError(f"Email {email} already in use")
            user.email = email
        if phone is not None:
            user.phone = phone
        if department is not None:
            user.department = department
        if role is not None:
            if role not in ["super_admin", "admin"]:
                raise ValueError("Role must be 'super_admin' or 'admin'")
            user.role = role
        if zone_access is not None:
            user.zone_access = zone_access
        if is_active is not None:
            user.is_active = is_active
        if password is not None:
            user.password_hash = hash_password(password)
        
        user.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(user)
        
        print(f"✅ User updated: {user.email}")
        return user
    
    except Exception as e:
        db.rollback()
        print(f"❌ Error updating user: {e}")
        raise


def delete_user(db: Session, user_id: int) -> bool:
    """Delete a user"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            return False
        
        db.delete(user)
        db.commit()
        
        print(f"✅ User deleted: {user.email}")
        return True
    
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting user: {e}")
        return False


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password"""
    try:
        user = get_user_by_email(db, email)
        
        if not user:
            print(f"❌ User not found: {email}")
            return None
        
        if not user.is_active:
            print(f"❌ User inactive: {email}")
            return None
        
        if not verify_password(password, user.password_hash):
            print(f"❌ Invalid password for: {email}")
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        print(f"✅ User authenticated: {email}")
        return user
    
    except Exception as e:
        print(f"❌ Error authenticating user: {e}")
        return None


def get_user_stats(db: Session) -> dict:
    """Get user statistics"""
    try:
        total = db.query(User).count()
        active = db.query(User).filter(User.is_active == True).count()
        super_admins = db.query(User).filter(User.role == "super_admin").count()
        admins = db.query(User).filter(User.role == "admin").count()
        
        return {
            "total": total,
            "active": active,
            "inactive": total - active,
            "super_admins": super_admins,
            "admins": admins
        }
    
    except Exception as e:
        print(f"❌ Error getting user stats: {e}")
        return {
            "total": 0,
            "active": 0,
            "inactive": 0,
            "super_admins": 0,
            "admins": 0
        }


# ======================================================
# ANALYTICS FUNCTIONS
# ======================================================

def get_violation_analysis(db: Session, start_date: datetime, end_date: datetime):
    """
    Get violation breakdown with counts and percentages
    
    Args:
        db: Database session
        start_date: Start of date range
        end_date: End of date range
    
    Returns:
        Dictionary with violation statistics
    """
    try:
        # Query to get violation counts by type
        violation_counts = (
            db.query(
                Violation.violation_type,
                func.count(Violation.id).label("count")
            )
            .filter(
                Violation.timestamp >= start_date,
                Violation.timestamp < end_date
            )
            .group_by(Violation.violation_type)
            .all()
        )
        
        # Calculate total incidents
        total_incidents = sum(count for _, count in violation_counts)
        
        # Format results with percentages
        violations = []
        for violation_type, count in violation_counts:
            percentage = (count / total_incidents * 100) if total_incidents > 0 else 0
            
            violations.append({
                "name": format_violation_name(violation_type),
                "type": violation_type,
                "incidents": count,
                "percentage": round(percentage, 1)
            })
        
        # Sort by count descending
        violations.sort(key=lambda x: x["incidents"], reverse=True)
        
        return {
            "violations": violations,
            "total_incidents": total_incidents
        }
    
    except Exception as e:
        print(f"Error in get_violation_analysis: {e}")
        return {
            "violations": [],
            "total_incidents": 0
        }


def format_violation_name(violation_type: str) -> str:
    """
    Convert violation type code to human-readable name
    
    Args:
        violation_type: Violation type code (e.g., 'no_helmet')
    
    Returns:
        Formatted name (e.g., 'No Helmet')
    """
    name_mapping = {
        "no_helmet": "No Helmet",
        "no_gloves": "No Gloves",
        "no_goggle": "No Goggles",
        "no_boots": "No Boots",
        "mobile_phone": "Mobile Phone Use",
        "fall": "Fall Detected",
        "Restricted Area": "Restricted Area Access",
        "fire": "Fire Detected",
        "smoke": "Smoke Detected"
    }
    
    return name_mapping.get(violation_type, violation_type.replace("_", " ").title())


def get_compliance_trend_analytics(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    filter_type: str
):
    """
    Get compliance rate trend over time (for Analytics tab)
    
    Args:
        db: Database session
        start_date: Start of period
        end_date: End of period
        filter_type: 'weekly' or 'monthly'
    
    Returns:
        List of compliance data points
    """
    try:
        if filter_type == "weekly":
            # Group by day for weekly view
            interval_days = 1
        else:  # monthly
            # Group by week for monthly view
            interval_days = 7
        
        compliance_data = []
        current_date = start_date
        
        while current_date < end_date:
            period_end = min(current_date + timedelta(days=interval_days), end_date)
            
            # Count violations in this period
            violation_count = (
                db.query(func.count(Violation.id))
                .filter(
                    Violation.timestamp >= current_date,
                    Violation.timestamp < period_end
                )
                .scalar()
            )
            
            # Estimate compliance rate
            baseline_checks = 100 * interval_days
            compliance_rate = max(0, ((baseline_checks - violation_count) / baseline_checks) * 100)
            
            compliance_data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "compliance_rate": round(compliance_rate, 1),
                "violations": violation_count
            })
            
            current_date = period_end
        
        return compliance_data
    
    except Exception as e:
        print(f"Error in get_compliance_trend_analytics: {e}")
        return []


def get_hourly_distribution(
    db: Session,
    start_date: datetime,
    end_date: datetime
):
    """
    Get incident count by hour of day (0-23)
    
    Args:
        db: Database session
        start_date: Start of day
        end_date: End of day
    
    Returns:
        List of hourly counts
    """
    try:
        # Query violations grouped by hour
        hourly_counts = (
            db.query(
                extract('hour', Violation.timestamp).label('hour'),
                func.count(Violation.id).label('count')
            )
            .filter(
                Violation.timestamp >= start_date,
                Violation.timestamp < end_date
            )
            .group_by(extract('hour', Violation.timestamp))
            .all()
        )
        
        # Create array with all 24 hours (0-23)
        hourly_data = [{"hour": hour, "count": 0} for hour in range(24)]
        
        # Fill in actual counts
        for hour, count in hourly_counts:
            if 0 <= int(hour) < 24:
                hourly_data[int(hour)]["count"] = count
        
        return hourly_data
    
    except Exception as e:
        print(f"Error in get_hourly_distribution: {e}")
        return [{"hour": hour, "count": 0} for hour in range(24)]


def get_zone_performance(db: Session, start_date: datetime, end_date: datetime):
    """
    Get zone performance metrics
    
    Args:
        db: Database session
        start_date: Start of period
        end_date: End of period
    
    Returns:
        Dictionary with zone scores, details, and incident counts
    """
    try:
        # Define zone mappings (handle different possible zone names in DB)
        zone_mappings = {
            "Zone A": ["Zone A", "ZONE A", "zone a", "ZONE_A", "zone_a", "A"],
            "Zone B": ["Zone B", "ZONE B", "zone b", "ZONE_B", "zone_b", "B"],
            "Zone C": ["Zone C", "ZONE C", "zone c", "ZONE_C", "zone_c", "C"],
            "Zone D": ["Zone D", "ZONE D", "zone d", "ZONE_D", "zone_d", "D"]
        }
        
        # Static worker counts per zone
        zone_workers = {
            "Zone A": 245,
            "Zone B": 189,
            "Zone C": 203,
            "Zone D": 210
        }
        
        zone_scores = []
        zone_details = []
        zone_incidents = []
        
        # First, get ALL violations in the date range to see what zones exist
        all_violations = (
            db.query(Violation.zone, func.count(Violation.id).label('count'))
            .filter(
                Violation.timestamp >= start_date,
                Violation.timestamp < end_date
            )
            .group_by(Violation.zone)
            .all()
        )
        
        print(f"\n=== DEBUG: Violations by zone in date range ===")
        print(f"Date range: {start_date} to {end_date}")
        for zone, count in all_violations:
            print(f"Zone: '{zone}' -> {count} incidents")
        
        # Process each zone
        for zone_name, zone_variants in zone_mappings.items():
            # Count incidents for this zone (check all variants)
            incident_count = 0
            
            for variant in zone_variants:
                count = (
                    db.query(func.count(Violation.id))
                    .filter(
                        Violation.zone == variant,
                        Violation.timestamp >= start_date,
                        Violation.timestamp < end_date
                    )
                    .scalar() or 0
                )
                incident_count += count
            
            print(f"Zone {zone_name}: {incident_count} total incidents")
            
            # Calculate safety score (0-100)
            safety_score = max(0, min(100, 100 - (incident_count * 5)))
            
            # Calculate risk score (0-10)
            risk_score = min(10.0, incident_count / 2.0)
            
            # Calculate compliance percentage
            workers = zone_workers.get(zone_name, 200)
            if incident_count == 0:
                compliance_percent = 100
            else:
                compliance_percent = max(0, 100 - (incident_count / workers * 100))
            
            # Determine badge class based on compliance
            if compliance_percent >= 90:
                badge_class = "bg-black text-white shadow-sm"
            elif compliance_percent >= 80:
                badge_class = "bg-gray-100 text-gray-900 border border-gray-200"
            else:
                badge_class = "bg-red-600 text-white shadow-sm"
            
            # Zone scores for radar chart
            zone_scores.append({
                "zone": zone_name,
                "score": round(safety_score, 1)
            })
            
            # Zone details for cards
            zone_details.append({
                "name": zone_name,
                "incidents": incident_count,
                "workers": workers,
                "risk": f"{risk_score:.1f}/10",
                "status": f"{int(compliance_percent)}% compliant",
                "badgeClass": badge_class
            })
            
            # Zone incidents for bar chart
            zone_incidents.append({
                "zone": zone_name,
                "value": incident_count
            })
        
        return {
            "zone_scores": zone_scores,
            "zone_details": zone_details,
            "zone_incidents": zone_incidents
        }
    
    except Exception as e:
        print(f"Error in get_zone_performance: {e}")
        import traceback
        traceback.print_exc()
        return {
            "zone_scores": [],
            "zone_details": [],
            "zone_incidents": []
        }