# api.py

import io
import json
import time
import asyncio
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional

from datetime import datetime, timedelta
from typing import Optional
from fastapi import Query

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, Depends, Query
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Add this import at the top
from datetime import datetime, timedelta
from typing import Optional

# Routers
from settings_manager import router as settings_router

# Worker Imports
from detector import YoloDetectorWorker
from email_worker import EmailWorker

# Database Imports
from database import get_db, engine
from models import Base, Violation
import crud

from typing import List

import secrets
from typing import Dict
# ======================================================
# FASTAPI SETUP
# ======================================================
app = FastAPI(title="Helmet AI API")
app.include_router(settings_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# WEBSOCKET SYSTEM
# ======================================================
active_connections = []


@app.websocket("/ws/alerts")
async def alert_websocket(ws: WebSocket):
    await ws.accept()
    active_connections.append(ws)
    try:
        while True:
            await ws.receive_text()
    except:
        if ws in active_connections:
            active_connections.remove(ws)


async def broadcast_alert(alert: dict):
    message = json.dumps(alert)
    dead_sockets = []

    for ws in list(active_connections):
        try:
            await ws.send_text(message)
        except:
            dead_sockets.append(ws)

    for ws in dead_sockets:
        if ws in active_connections:
            active_connections.remove(ws)


# ======================================================
# NOTIFICATION STORAGE (In-Memory) - Keep for frontend
# ======================================================
notifications_store = deque(maxlen=100)
notification_id_counter = 0


# Find create_notification function and update it:
def create_notification(stream_id: str, violation: str, zone_id: str):
    """Create a notification object for frontend"""
    global notification_id_counter
    notification_id_counter += 1
    
    # Determine severity - ✅ UPDATED: Added "group_monitoring"
    if violation in ["no_helmet", "fall", "restricted_area", "fire", "dwell", "group_monitoring"]:
        severity = "critical"
    elif violation in ["no_goggle", "no_gloves", "no_boots", "mobile_phone", "smoke"]:
        severity = "warning"
    else:
        severity = "info"
    
    # ✅ UPDATED: Added "group_monitoring" to title mapping
    violation_titles = {
        "no_helmet": "Critical Safety Violation Detected",
        "no_goggle": "PPE Violation: Goggles Missing",
        "no_gloves": "PPE Violation: Gloves Missing",
        "no_boots": "PPE Violation: Boots Missing",
        "fall": "CRITICAL: Fall Incident Detected",
        "mobile_phone": "Unauthorized Mobile Phone Use",
        "restricted_area": "Restricted Area Violation",
        "fire": "🔥 CRITICAL FIRE ALERT",
        "smoke": "Smoke Detected - Warning",
        "dwell": "⏱️ CRITICAL: Excessive Dwell Time Detected",
        "group_monitoring": "👥 CRITICAL: Multiple Persons Detected"  # ✅ NEW
    }
    
    # ✅ UPDATED: Added "group_monitoring" to description mapping
    violation_descriptions = {
        "no_helmet": f"Worker without helmet detected in {zone_id} - Stream {stream_id}",
        "no_goggle": f"Worker without safety goggles in {zone_id} - Stream {stream_id}",
        "no_gloves": f"Worker without safety gloves in {zone_id} - Stream {stream_id}",
        "no_boots": f"Worker without safety boots in {zone_id} - Stream {stream_id}",
        "fall": f"Fall incident detected in {zone_id} - Stream {stream_id}. Immediate action required!",
        "mobile_phone": f"Unauthorized mobile phone usage detected in {zone_id} - Stream {stream_id}",
        "restricted_area": f"Person entered restricted area in {zone_id} - Stream {stream_id}",
        "fire": f"🔥 FIRE DETECTED in {zone_id} - Stream {stream_id}. Emergency response required!",
        "smoke": f"Smoke detected in {zone_id} - Stream {stream_id}. Please investigate.",
        "dwell": f"Person exceeded 4-second dwell time in {zone_id} - Stream {stream_id}. Potential safety concern!",
        "group_monitoring": f"Multiple persons detected in restricted area {zone_id} - Stream {stream_id}. Only single person allowed!"  # ✅ NEW
    }
    
    notification = {
        "id": notification_id_counter,
        "type": severity,
        "title": violation_titles.get(violation, f"Alert: {violation}"),
        "description": violation_descriptions.get(violation, f"Violation '{violation}' detected"),
        "timestamp": datetime.now().isoformat(),
        "stream_id": stream_id,
        "zone": zone_id,
        "violation": violation,
        "source": "AI Detection",
        "read": False
    }
    
    notifications_store.appendleft(notification)
    return notification



# ======================================================
# ZONE CONFIGURATION
# ======================================================
# Find ZONE_RULES section and replace with this:
ZONE_RULES = {
    "ZONE A": {
        "no_helmet": False,
        "no_gloves": False,
        "no_goggle": False,
        "no_boots": False,
        "mobile_phone": False,
        "fall": False,
        "Restricted Area": False,
        "fire": False,
        "smoke": False,
        "Dwell": False,
        "Group_Monitoring": True  # ✅ NEW
    },
    "ZONE B": {
        "no_helmet": True,
        "no_gloves": True,
        "no_goggle": True,
        "no_boots": True,
        "mobile_phone": True,
        "fall": True,
        "Restricted Area": False,
        "fire": True,
        "smoke": True,
        "Dwell": True,
        "Group_Monitoring": True  # ✅ NEW
    },
    "ZONE C": {
        "no_helmet": True,
        "no_gloves": True,
        "no_goggle": True,
        "no_boots": True,
        "mobile_phone": True,
        "fall": True,
        "Restricted Area": False,
        "fire": False,
        "smoke": False,
        "Dwell": True,
        "Group_Monitoring": True  # ✅ NEW
    },
    "ZONE D": {
        "no_helmet": False,
        "no_gloves": False,
        "no_goggle": False,
        "no_boots": False,
        "mobile_phone": False,
        "fall": False,
        "Restricted Area": True,
        "fire": False,
        "smoke": False,
        "Dwell": True,
        "Group_Monitoring": True  # ✅ NEW
    },
    "Break Room": {
        "no_helmet": False,
        "no_gloves": False,
        "no_goggle": False,
        "no_boots": False,
        "mobile_phone": False,
        "fall": False,
        "Restricted Area": False,
        "fire": True,
        "smoke": True,
        "Dwell": True,
        "Group_Monitoring": True  # ✅ NEW
    },
}

DEFAULT_ZONE = "ZONE A"


# ======================================================
# LOGGING
# ======================================================
LOG_FILE = "alerts.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("🚀 Helmet AI API started")


# ======================================================
# GLOBALS
# ======================================================
MP_MANAGER = None
SHARED_FRAMES = None
STOP_EVENT = None
ALERT_QUEUE = None
EMAIL_SENDER_THREAD = None

ACTIVE_WORKERS = {}
STREAM_ZONES = {}


# ======================================================
# MODELS
# ======================================================
class StreamSource(BaseModel):
    source: str
    zone_id: str = DEFAULT_ZONE


class AlertRequest(BaseModel):
    stream_id: str
    violation: str


class ViolationStatusUpdate(BaseModel):
    status: str


# ======================================================
# STARTUP & SHUTDOWN
# ======================================================
@app.on_event("startup")
def startup_event():
    global MP_MANAGER, SHARED_FRAMES, STOP_EVENT, ALERT_QUEUE, EMAIL_SENDER_THREAD

    # Create database tables
    print("🗄️  Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables ready")

    MP_MANAGER = mp.Manager()
    SHARED_FRAMES = MP_MANAGER.dict()
    STOP_EVENT = mp.Event()
    ALERT_QUEUE = MP_MANAGER.Queue()

    EMAIL_SENDER_THREAD = EmailWorker(ALERT_QUEUE, STOP_EVENT)
    EMAIL_SENDER_THREAD.start()

    print("[API] Manager & EmailWorker initialized.")
    logging.info("[SYSTEM] Startup complete.")


@app.on_event("shutdown")
def shutdown_event():
    global STOP_EVENT, EMAIL_SENDER_THREAD, MP_MANAGER

    if STOP_EVENT:
        STOP_EVENT.set()

    if EMAIL_SENDER_THREAD and EMAIL_SENDER_THREAD.is_alive():
        EMAIL_SENDER_THREAD.join(timeout=1)

    if MP_MANAGER:
        try:
            MP_MANAGER.shutdown()
        except:
            pass

    print("[API] Shutdown complete.")
    logging.info("[SYSTEM] Shutdown complete.")


# ======================================================
# STREAM MANAGEMENT
# ======================================================
@app.get("/streams")
def list_streams():
    return {
        "streams": {
            sid: STREAM_ZONES.get(sid, DEFAULT_ZONE)
            for sid in ACTIVE_WORKERS.keys()
        }
    }


@app.post("/streams")
def add_stream(data: StreamSource):
    global ACTIVE_WORKERS, STREAM_ZONES

    stream_id = Path(data.source).name if not data.source.isdigit() else data.source
    zone_id = data.zone_id if data.zone_id in ZONE_RULES else DEFAULT_ZONE

    if stream_id in ACTIVE_WORKERS and ACTIVE_WORKERS[stream_id].is_alive():
        return {
            "message": f"Stream {stream_id} already active in {STREAM_ZONES.get(stream_id)}"
        }

    worker = YoloDetectorWorker(
        stream_id,
        data.source,
        SHARED_FRAMES,
        STOP_EVENT,
        ALERT_QUEUE,
        zone_rules=ZONE_RULES,
        zone_id=zone_id,
    )
    worker.start()

    ACTIVE_WORKERS[stream_id] = worker
    STREAM_ZONES[stream_id] = zone_id

    print(f"[API] Stream started → {stream_id}  Zone: {zone_id}")
    logging.info(f"[STREAM] Started {stream_id} in {zone_id}")

    return {"message": "Stream started", "stream_id": stream_id, "zone_id": zone_id}


@app.delete("/streams/{stream_id}")
def stop_stream(stream_id: str):
    if stream_id not in ACTIVE_WORKERS:
        raise HTTPException(status_code=404, detail="Stream not found")

    worker = ACTIVE_WORKERS[stream_id]
    try:
        worker.terminate()
        worker.join(timeout=0.5)
    except:
        pass

    del ACTIVE_WORKERS[stream_id]
    STREAM_ZONES.pop(stream_id, None)

    print(f"[API] Stream stopped → {stream_id}")
    logging.info(f"[STREAM] Stopped {stream_id}")

    return {"message": "Stream stopped"}


@app.post("/streams/stop_all")
def stop_all_streams():
    for worker in list(ACTIVE_WORKERS.values()):
        try:
            worker.terminate()
            worker.join(timeout=0.5)
        except:
            pass

    ACTIVE_WORKERS.clear()
    STREAM_ZONES.clear()

    print("[API] All streams stopped")
    logging.info("[STREAM] All streams stopped")

    return {"message": "All streams stopped"}


# ======================================================
# SNAPSHOT ENDPOINT
# ======================================================
@app.get("/streams/{stream_id}/snapshot")
def get_snapshot(stream_id: str):
    if stream_id not in SHARED_FRAMES:
        raise HTTPException(status_code=404, detail="Frame not found")

    frame, _ = SHARED_FRAMES[stream_id]
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Encoding error")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
    )


# ======================================================
# NOTIFICATION ENDPOINTS (Frontend compatibility)
# ======================================================
@app.get("/notifications")
def get_notifications():
    """Get all notifications (in-memory for frontend)"""
    return {"notifications": list(notifications_store)}


@app.get("/notifications/stats")
def get_notification_stats():
    """Get notification statistics (in-memory for frontend)"""
    total = len(notifications_store)
    critical = sum(1 for n in notifications_store if n["type"] == "critical")
    warnings = sum(1 for n in notifications_store if n["type"] == "warning")
    unread = sum(1 for n in notifications_store if not n["read"])
    
    return {
        "total": total,
        "critical": critical,
        "warnings": warnings,
        "unread": unread
    }


@app.post("/notifications/{notification_id}/mark-read")
def mark_notification_read(notification_id: int):
    """Mark a notification as read"""
    for notification in notifications_store:
        if notification["id"] == notification_id:
            notification["read"] = True
            return {"status": "success"}
    
    raise HTTPException(status_code=404, detail="Notification not found")


@app.delete("/notifications/{notification_id}")
def delete_notification(notification_id: int):
    """Delete a notification"""
    global notifications_store
    notifications_store = deque(
        [n for n in notifications_store if n["id"] != notification_id],
        maxlen=100
    )
    return {"status": "success"}


# ======================================================
# DATABASE VIOLATION ENDPOINTS
# ======================================================
@app.get("/violations")
def get_violations_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    violation_type: Optional[str] = None,
    zone: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get violations from database with filters"""
    violations = crud.get_violations(
        db=db,
        skip=skip,
        limit=limit,
        violation_type=violation_type,
        zone=zone,
        severity=severity,
        status=status
    )
    
    return {
        "violations": [v.to_dict() for v in violations],
        "count": len(violations)
    }


@app.get("/violations/{violation_id}")
def get_violation_endpoint(violation_id: int, db: Session = Depends(get_db)):
    """Get specific violation by ID"""
    violation = crud.get_violation(db, violation_id)
    
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    return violation.to_dict()


@app.get("/violations/{violation_id}/image")
def get_violation_image(violation_id: int, db: Session = Depends(get_db)):
    """Get violation image"""
    violation = crud.get_violation(db, violation_id)
    
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    if not violation.photo_path or not Path(violation.photo_path).exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(violation.photo_path, media_type="image/jpeg")


@app.patch("/violations/{violation_id}/status")
def update_violation_status_endpoint(
    violation_id: int,
    data: ViolationStatusUpdate,
    db: Session = Depends(get_db)
):
    """Update violation status"""
    violation = crud.update_violation_status(db, violation_id, data.status)
    
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    return {"status": "success", "violation": violation.to_dict()}


@app.delete("/violations/{violation_id}")
def delete_violation_endpoint(violation_id: int, db: Session = Depends(get_db)):
    """Delete violation"""
    success = crud.delete_violation(db, violation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    return {"status": "success"}


@app.get("/violations/stats/summary")
def get_violation_stats_endpoint(db: Session = Depends(get_db)):
    """Get violation statistics"""
    stats = crud.get_violation_stats(db)
    by_zone = crud.get_violations_by_zone(db)
    by_type = crud.get_violations_by_type(db)
    
    return {
        "summary": stats,
        "by_zone": by_zone,
        "by_type": by_type
    }


# ======================================================
# REPORT ENDPOINTS
# ======================================================
@app.get("/reports/daily")
def get_daily_report_endpoint(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db)
):
    """Get daily violation report"""
    try:
        report = crud.get_daily_report(db, date)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error in daily report: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@app.get("/reports/weekly")
def get_weekly_report_endpoint(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    db: Session = Depends(get_db)
):
    """Get weekly violation report"""
    try:
        report = crud.get_weekly_report(db, start_date, end_date)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error in weekly report: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@app.get("/reports/monthly")
def get_monthly_report_endpoint(
    year: int = Query(..., description="Year (e.g., 2026)"),
    month: int = Query(..., ge=1, le=12, description="Month (1-12)"),
    db: Session = Depends(get_db)
):
    """Get monthly violation report"""
    try:
        report = crud.get_monthly_report(db, year, month)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error in monthly report: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@app.get("/reports/export/csv")
def export_report_csv(
    report_type: str = Query(..., description="daily, weekly, or monthly"),
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Export report as CSV"""
    try:
        # Get report data based on type
        if report_type == "daily":
            if not date:
                raise HTTPException(status_code=400, detail="Date required for daily report")
            report = crud.get_daily_report(db, date)
            filename = f"daily_report_{date}.csv"
            
        elif report_type == "weekly":
            if not start_date or not end_date:
                raise HTTPException(status_code=400, detail="Start and end dates required for weekly report")
            report = crud.get_weekly_report(db, start_date, end_date)
            filename = f"weekly_report_{start_date}_to_{end_date}.csv"
            
        elif report_type == "monthly":
            if not year or not month:
                raise HTTPException(status_code=400, detail="Year and month required for monthly report")
            report = crud.get_monthly_report(db, year, month)
            filename = f"monthly_report_{year}_{month:02d}.csv"
            
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")
        
        # Convert violations to CSV
        violations = [crud.get_violation(db, v["id"]) for v in report["violations"]]
        violations = [v for v in violations if v is not None]
        csv_content = crud.export_violations_to_csv(violations)
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")


# ======================================================
# ALERT ENDPOINTS (Modified to save to DB)
# ======================================================
# Find trigger_violation_alert function and update valid violations list:
@app.post("/alerts/{violation_name}", status_code=201)
def trigger_violation_alert(
    violation_name: str,
    data: AlertRequest,
    db: Session = Depends(get_db)
):
    # ✅ UPDATED: Added "group_monitoring" to valid violations list
    valid = [
        "no_helmet",
        "no_goggle",
        "no_gloves",
        "no_boots",
        "fall",
        "mobile_phone",
        "restricted_area",
        "fire",
        "smoke",
        "dwell",
        "group_monitoring"  # ✅ NEW
    ]

    if violation_name not in valid:
        raise HTTPException(status_code=400, detail="Invalid violation type")

    # Zone rule validation
    zone = STREAM_ZONES.get(data.stream_id, DEFAULT_ZONE)
    rules = ZONE_RULES.get(zone, {})

    # ✅ UPDATED: Added "group_monitoring" rule key mapping
    if violation_name == "restricted_area":
        rule_key = "Restricted Area"
    elif violation_name == "dwell":
        rule_key = "Dwell"
    elif violation_name == "group_monitoring":
        rule_key = "Group_Monitoring"  # ✅ NEW
    else:
        rule_key = violation_name

    if rules.get(rule_key) is not True:
        return {"status": "blocked_by_zone", "zone": zone}

    try:
        # Get frame
        if data.stream_id in SHARED_FRAMES:
            frame, _ = SHARED_FRAMES[data.stream_id]
        else:
            frame = 255 * np.ones((100, 100, 3), dtype=np.uint8)

        # ✅ UPDATED: Added "group_monitoring" to severity classification
        if violation_name in ["no_helmet", "fall", "restricted_area", "fire", "dwell", "group_monitoring"]:
            severity = "critical"
        elif violation_name in ["no_goggle", "no_gloves", "no_boots", "mobile_phone", "smoke"]:
            severity = "warning"
        else:
            severity = "info"

        # Save image to disk
        photo_path = crud.save_violation_image(frame, violation_name, data.stream_id)

        # ✅ UPDATED: Added "group_monitoring" to title mapping
        violation_titles = {
            "no_helmet": "Critical Safety Violation Detected",
            "no_goggle": "PPE Violation: Goggles Missing",
            "no_gloves": "PPE Violation: Gloves Missing",
            "no_boots": "PPE Violation: Boots Missing",
            "fall": "CRITICAL: Fall Incident Detected",
            "mobile_phone": "Unauthorized Mobile Phone Use",
            "restricted_area": "Restricted Area Violation",
            "fire": "🔥 CRITICAL FIRE ALERT",
            "smoke": "Smoke Detected - Warning",
            "dwell": "⏱️ CRITICAL: Excessive Dwell Time Detected",
            "group_monitoring": "👥 CRITICAL: Multiple Persons Detected"  # ✅ NEW
        }
        
        # ✅ UPDATED: Added "group_monitoring" to description mapping
        violation_descriptions = {
            "no_helmet": f"Worker without helmet detected in {zone} - Stream {data.stream_id}",
            "no_goggle": f"Worker without safety goggles in {zone} - Stream {data.stream_id}",
            "no_gloves": f"Worker without safety gloves in {zone} - Stream {data.stream_id}",
            "no_boots": f"Worker without safety boots in {zone} - Stream {data.stream_id}",
            "fall": f"Fall incident detected in {zone} - Stream {data.stream_id}",
            "mobile_phone": f"Unauthorized mobile phone usage in {zone} - Stream {data.stream_id}",
            "restricted_area": f"Person entered restricted area in {zone} - Stream {data.stream_id}",
            "fire": f"FIRE DETECTED in {zone} - Stream {data.stream_id}",
            "smoke": f"Smoke detected in {zone} - Stream {data.stream_id}",
            "dwell": f"Person exceeded 4-second dwell time in {zone} - Stream {data.stream_id}. Potential safety concern!",
            "group_monitoring": f"Multiple persons detected in {zone} - Stream {data.stream_id}. Only single person allowed!"  # ✅ NEW
        }

        # Save to database
        db_violation = crud.create_violation(
            db=db,
            violation_type=violation_name,
            stream_id=data.stream_id,
            zone=zone,
            severity=severity,
            photo_path=photo_path,
            title=violation_titles.get(violation_name, f"Alert: {violation_name}"),
            description=violation_descriptions.get(violation_name, f"Violation '{violation_name}' detected")
        )

        # Add to email queue
        ALERT_QUEUE.put((data.stream_id, violation_name, frame))

        # Create notification for frontend
        notification = create_notification(data.stream_id, violation_name, zone)

        log_msg = f"[ALERT] {violation_name.upper()} → Stream {data.stream_id} → DB ID: {db_violation.id}"
        print(f"[API] 🚨 {log_msg}")
        logging.info(log_msg)

        payload = {
            "stream_id": data.stream_id,
            "violation": violation_name,
            "zone_id": zone,
            "timestamp": time.time(),
            "notification": notification,
            "db_id": db_violation.id
        }

        # Broadcast WS alert
        try:
            anyio.from_thread.run(broadcast_alert, payload)
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                asyncio.create_task, broadcast_alert(payload)
            )

        return {
            "status": "success",
            "stream_id": data.stream_id,
            "violation": violation_name,
            "db_id": db_violation.id
        }

    except Exception as e:
        msg = f"[ERROR] Failed alert: {e}"
        print(msg)
        logging.error(msg)
        raise HTTPException(status_code=500, detail="Alert error")


# Dedicated Routes
@app.post("/alerts/no_helmet")
def no_helmet_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("no_helmet", data, db)


@app.post("/alerts/no_goggle")
def no_goggle_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("no_goggle", data, db)


@app.post("/alerts/no_gloves")
def no_gloves_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("no_gloves", data, db)


@app.post("/alerts/no_boots")
def no_boots_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("no_boots", data, db)


@app.post("/alerts/fall")
def fall_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("fall", data, db)


@app.post("/alerts/mobile_phone")
def mobile_phone_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("mobile_phone", data, db)


@app.post("/alerts/restricted_area")
def restricted_area_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("restricted_area", data, db)


@app.post("/alerts/fire")
def fire_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("fire", data, db)


@app.post("/alerts/smoke")
def smoke_alert(data: AlertRequest, db: Session = Depends(get_db)):
    return trigger_violation_alert("smoke", data, db)


@app.post("/alerts/dwell")
def dwell_alert(data: AlertRequest, db: Session = Depends(get_db)):
    """Handle dwell time violation alert - uses generic endpoint"""
    return trigger_violation_alert("dwell", data, db)

# Add dedicated endpoint for group_monitoring (after dwell_alert endpoint):
@app.post("/alerts/group_monitoring")
def group_monitoring_alert(data: AlertRequest, db: Session = Depends(get_db)):
    """Handle group monitoring violation alert"""
    return trigger_violation_alert("group_monitoring", data, db)


# ======================================================
# DASHBOARD ANALYTICS ENDPOINTS
# ======================================================

@app.get("/dashboard/compliance-trend")
def get_compliance_trend_endpoint(
    period: str = Query("weekly", description="daily, weekly, or monthly"),
    date: Optional[str] = Query(None, description="Reference date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get compliance trend data for dashboard chart"""
    try:
        if period not in ["daily", "weekly", "monthly"]:
            raise HTTPException(status_code=400, detail="Period must be 'daily', 'weekly', or 'monthly'")
        
        data = crud.get_compliance_trend(db, period, date)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error getting compliance trend: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting compliance trend: {str(e)}")


@app.get("/dashboard/top-violations")
def get_top_violations_endpoint(
    period: str = Query("weekly", description="daily, weekly, or monthly"),
    date: Optional[str] = Query(None, description="Reference date (YYYY-MM-DD)"),
    limit: int = Query(5, ge=1, le=10, description="Number of top violations"),
    db: Session = Depends(get_db)
):
    """Get top violations data for dashboard pie chart"""
    try:
        if period not in ["daily", "weekly", "monthly"]:
            raise HTTPException(status_code=400, detail="Period must be 'daily', 'weekly', or 'monthly'")
        
        data = crud.get_top_violations(db, period, date, limit)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error getting top violations: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting top violations: {str(e)}")


@app.get("/dashboard/recent-incidents")
def get_recent_incidents_endpoint(
    period: str = Query("all", description="all, daily, weekly, or monthly"),
    date: Optional[str] = Query(None, description="Reference date (YYYY-MM-DD)"),
    limit: int = Query(3, ge=1, le=100, description="Number of incidents per page"),
    skip: int = Query(0, ge=0, description="Number of incidents to skip (for pagination)"),
    db: Session = Depends(get_db)
):
    """Get recent incidents for dashboard with pagination"""
    try:
        if period not in ["all", "daily", "weekly", "monthly"]:
            raise HTTPException(
                status_code=400, 
                detail="Period must be 'all', 'daily', 'weekly', or 'monthly'"
            )
        
        result = crud.get_recent_incidents(db, period, date, limit, skip)
        
        return {
            "incidents": result["incidents"],
            "total": result["total"],
            "limit": result["limit"],
            "skip": result["skip"],
            "period": period
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error getting recent incidents: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting recent incidents: {str(e)}"
        )


@app.get("/dashboard/zone-risk-status")
def get_zone_risk_status_endpoint(
    period: str = Query("weekly", description="daily, weekly, or monthly"),
    date: Optional[str] = Query(None, description="Reference date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get zone risk status for dashboard"""
    try:
        if period not in ["daily", "weekly", "monthly"]:
            raise HTTPException(
                status_code=400, 
                detail="Period must be 'daily', 'weekly', or 'monthly'"
            )
        
        zones = crud.get_zone_risk_status(db, period, date)
        return {"zones": zones, "period": period}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error getting zone risk status: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting zone risk status: {str(e)}"
        )

# ======================================================
# PYDANTIC MODELS FOR USER ENDPOINTS
# ======================================================

class UserCreate(BaseModel):
    full_name: str
    email: str
    password: str
    phone: Optional[str] = None
    department: Optional[str] = None
    role: str
    zone_access: Optional[List[str]] = []


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    zone_access: Optional[List[str]] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


# ======================================================
# USER MANAGEMENT ENDPOINTS (NO JWT)
# ======================================================

@app.post("/auth/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """Simple login - returns user data (NO JWT)"""
    try:
        user = crud.authenticate_user(db, data.email, data.password)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        return {
            "status": "success",
            "user": user.to_dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Login error")


@app.get("/users")
def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all users"""
    try:
        users = crud.get_all_users(db, skip, limit, role, is_active)
        return {
            "users": [u.to_dict() for u in users],
            "count": len(users)
        }
    except Exception as e:
        print(f"❌ Error getting users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/stats")
def get_users_stats(db: Session = Depends(get_db)):
    """Get user statistics"""
    try:
        stats = crud.get_user_stats(db)
        return stats
    except Exception as e:
        print(f"❌ Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID"""
    try:
        user = crud.get_user_by_id(db, user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users", status_code=201)
def create_user_endpoint(data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        user = crud.create_user(
            db=db,
            full_name=data.full_name,
            email=data.email,
            password=data.password,
            role=data.role,
            phone=data.phone,
            department=data.department,
            zone_access=data.zone_access
        )
        
        return {
            "status": "success",
            "user": user.to_dict()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/users/{user_id}")
def update_user_endpoint(
    user_id: int,
    data: UserUpdate,
    db: Session = Depends(get_db)
):
    """Update user details"""
    try:
        user = crud.update_user(
            db=db,
            user_id=user_id,
            full_name=data.full_name,
            email=data.email,
            phone=data.phone,
            department=data.department,
            role=data.role,
            zone_access=data.zone_access,
            is_active=data.is_active,
            password=data.password
        )
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "status": "success",
            "user": user.to_dict()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating user: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}")
def delete_user_endpoint(user_id: int, db: Session = Depends(get_db)):
    """Delete a user"""
    try:
        success = crud.delete_user(db, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"status": "success"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# ANALYTICS ENDPOINTS
# ======================================================

@app.get("/analytics/violation-analysis")
def get_violation_analysis(
    filter_type: str = Query(..., description="Filter type: daily, weekly, or monthly"),
    date: Optional[str] = Query(None, description="Date for daily filter (YYYY-MM-DD)"),
    start_date: Optional[str] = Query(None, description="Start date for weekly filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for weekly filter (YYYY-MM-DD)"),
    month: Optional[int] = Query(None, description="Month for monthly filter (1-12)"),
    year: Optional[int] = Query(None, description="Year for monthly filter (e.g., 2026)"),
    db: Session = Depends(get_db)
):
    """Get violation breakdown and statistics with flexible date filtering"""
    try:
        if filter_type == "daily":
            if not date:
                return {"error": "Date parameter is required for daily filter"}
            
            reference_date = datetime.strptime(date, "%Y-%m-%d")
            start = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        
        elif filter_type == "weekly":
            if not start_date or not end_date:
                return {"error": "start_date and end_date parameters are required for weekly filter"}
            
            start = datetime.strptime(start_date, "%Y-%m-%d")
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = datetime.strptime(end_date, "%Y-%m-%d")
            end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif filter_type == "monthly":
            if not month or not year:
                return {"error": "month and year parameters are required for monthly filter"}
            
            if month < 1 or month > 12:
                return {"error": "Month must be between 1 and 12"}
            
            start = datetime(year, month, 1, 0, 0, 0, 0)
            
            if month == 12:
                end = datetime(year + 1, 1, 1, 0, 0, 0, 0)
            else:
                end = datetime(year, month + 1, 1, 0, 0, 0, 0)
        
        else:
            return {"error": "Invalid filter_type. Use: daily, weekly, or monthly"}
        
        violation_stats = crud.get_violation_analysis(
            db=db,
            start_date=start,
            end_date=end
        )
        
        return {
            "filter_type": filter_type,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "violations": violation_stats["violations"],
            "total_incidents": violation_stats["total_incidents"]
        }
    
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/compliance-trend")
def get_compliance_trend_analytics_endpoint(
    filter_type: str = Query(...),
    start_date: Optional[str] = Query(None, description="Start date for weekly filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for weekly filter (YYYY-MM-DD)"),
    month: Optional[int] = Query(None, description="Month for monthly filter (1-12)"),
    year: Optional[int] = Query(None, description="Year for monthly filter"),
    db: Session = Depends(get_db)
):
    """Get compliance rate trend over time"""
    try:
        if filter_type == "weekly":
            if not start_date or not end_date:
                return {"error": "start_date and end_date required for weekly filter"}
            
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        
        elif filter_type == "monthly":
            if not month or not year:
                return {"error": "month and year required for monthly filter"}
            
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
        
        else:
            return {"error": "Invalid filter_type. Use: weekly or monthly"}
        
        compliance_data = crud.get_compliance_trend_analytics(
            db=db,
            start_date=start,
            end_date=end,
            filter_type=filter_type
        )
        
        return {
            "filter_type": filter_type,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "data": compliance_data
        }
    
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/hourly-distribution")
def get_hourly_distribution(
    date: str = Query(..., description="Date for hourly distribution (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get incident distribution by hour for a specific day"""
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        hourly_data = crud.get_hourly_distribution(
            db=db,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "date": date,
            "data": hourly_data
        }
    
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics/zone-performance")
def get_zone_performance(
    filter_type: str = Query(..., description="Filter type: daily, weekly, or monthly"),
    date: Optional[str] = Query(None, description="Date for daily filter (YYYY-MM-DD)"),
    start_date: Optional[str] = Query(None, description="Start date for weekly filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for weekly filter (YYYY-MM-DD)"),
    month: Optional[int] = Query(None, description="Month for monthly filter (1-12)"),
    year: Optional[int] = Query(None, description="Year for monthly filter"),
    db: Session = Depends(get_db)
):
    """Get zone performance metrics"""
    try:
        if filter_type == "daily":
            if not date:
                return {"error": "date parameter required for daily filter"}
            
            reference_date = datetime.strptime(date, "%Y-%m-%d")
            start = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        
        elif filter_type == "weekly":
            if not start_date or not end_date:
                return {"error": "start_date and end_date required for weekly filter"}
            
            start = datetime.strptime(start_date, "%Y-%m-%d")
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = datetime.strptime(end_date, "%Y-%m-%d")
            end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif filter_type == "monthly":
            if not month or not year:
                return {"error": "month and year required for monthly filter"}
            
            start = datetime(year, month, 1, 0, 0, 0, 0)
            if month == 12:
                end = datetime(year + 1, 1, 1, 0, 0, 0, 0)
            else:
                end = datetime(year, month + 1, 1, 0, 0, 0, 0)
        
        else:
            return {"error": "Invalid filter_type. Use: daily, weekly, or monthly"}
        
        zone_data = crud.get_zone_performance(
            db=db,
            start_date=start,
            end_date=end
        )
        
        return {
            "filter_type": filter_type,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "zone_scores": zone_data["zone_scores"],
            "zone_details": zone_data["zone_details"],
            "zone_incidents": zone_data["zone_incidents"]
        }
    
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}
    

# Add after existing imports in api.py
from typing import Dict
import secrets

# Simple in-memory storage
TEMP_TOKENS: Dict[str, dict] = {}
ACTIVE_SESSIONS: Dict[str, dict] = {}

# ======================================================
# PYDANTIC MODELS FOR SIMPLE AUTH
# ======================================================
class SimpleLoginRequest(BaseModel):
    email: str
    password: str

class SimpleVerify2FARequest(BaseModel):
    temp_token: str
    code: str

class SimpleLogoutRequest(BaseModel):
    session_id: str

# ======================================================
# SIMPLE 2FA AUTHENTICATION ENDPOINTS
# ======================================================

@app.post("/auth/simple-login")
def simple_login(data: SimpleLoginRequest, db: Session = Depends(get_db)):
    """
    Step 1: Validate credentials and return temp token
    """
    try:
        # Authenticate user
        user = crud.authenticate_user(db, data.email, data.password)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Generate simple temp token
        temp_token = secrets.token_urlsafe(16)
        
        # Store temp token (expires in 5 minutes)
        TEMP_TOKENS[temp_token] = {
            "user_id": user.id,
            "email": user.email,
            "role": user.role,
            "full_name": user.full_name,
            "created_at": time.time()
        }
        
        # Clean expired tokens
        current_time = time.time()
        expired = [t for t, d in TEMP_TOKENS.items() if current_time - d["created_at"] > 300]
        for t in expired:
            del TEMP_TOKENS[t]
        
        print(f"✅ Login success: {user.email}")
        
        return {
            "success": True,
            "temp_token": temp_token,
            "message": "Please enter 2FA code"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/auth/simple-verify-2fa")
def simple_verify_2fa(data: SimpleVerify2FARequest):
    """
    Step 2: Verify 2FA and return user data
    ✅ ACCEPTS ANY 6-DIGIT CODE - NO VALIDATION
    """
    try:
        print(f"🔐 2FA Request - Token: {data.temp_token}, Code: {data.code}")
        
        # Check temp token exists
        if data.temp_token not in TEMP_TOKENS:
            print(f"❌ Temp token not found: {data.temp_token}")
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        temp_data = TEMP_TOKENS[data.temp_token]
        
        # Check expiry (5 minutes)
        if time.time() - temp_data["created_at"] > 300:
            del TEMP_TOKENS[data.temp_token]
            print(f"❌ Token expired for: {temp_data['email']}")
            raise HTTPException(status_code=401, detail="Token expired. Please login again.")
        
        # ✅ SIMPLE VALIDATION: Just check if code is 6 digits
        # NO ACTUAL VERIFICATION - ACCEPT ANY 6 DIGITS
        if not data.code or len(data.code) != 6 or not data.code.isdigit():
            print(f"❌ Invalid code format: {data.code}")
            raise HTTPException(status_code=401, detail="Please enter exactly 6 digits")
        
        # ✅ CODE IS VALID (any 6 digits) - Generate session
        session_id = secrets.token_urlsafe(16)
        
        # Store active session
        ACTIVE_SESSIONS[session_id] = {
            "user_id": temp_data["user_id"],
            "email": temp_data["email"],
            "role": temp_data["role"],
            "full_name": temp_data["full_name"],
            "created_at": time.time()
        }
        
        # Delete temp token
        del TEMP_TOKENS[data.temp_token]
        
        print(f"✅ 2FA SUCCESS: {temp_data['email']} - Code: {data.code}")
        
        return {
            "success": True,
            "session_id": session_id,
            "user": {
                "id": temp_data["user_id"],
                "email": temp_data["email"],
                "role": temp_data["role"],
                "full_name": temp_data["full_name"]
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 2FA error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Verification failed")


@app.get("/auth/simple-verify-session")
def simple_verify_session(session_id: str = Query(..., description="Session ID")):
    """
    Verify if session is valid
    """
    try:
        if session_id not in ACTIVE_SESSIONS:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        session_data = ACTIVE_SESSIONS[session_id]
        
        return {
            "success": True,
            "user": {
                "id": session_data["user_id"],
                "email": session_data["email"],
                "role": session_data["role"],
                "full_name": session_data["full_name"]
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Session invalid")


@app.post("/auth/simple-logout")
def simple_logout(data: SimpleLogoutRequest):
    """
    Logout and clear session
    """
    try:
        if data.session_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[data.session_id]
            print(f"✅ Session logged out: {data.session_id[:10]}...")
        
        return {"success": True}
    
    except Exception as e:
        print(f"❌ Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")