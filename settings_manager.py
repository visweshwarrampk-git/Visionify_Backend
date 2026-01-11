# settings_manager.py
from fastapi import APIRouter
from pydantic import BaseModel
import json, os

router = APIRouter()

SETTINGS_FILE = "alert_settings.json"


class AlertChannels(BaseModel):
    email: bool
    sms: bool
    push: bool
    sound: bool


class AlertSettings(BaseModel):
    critical: AlertChannels
    warning: AlertChannels
    info: AlertChannels


@router.post("/settings/alerts")
def save_alert_settings(settings: AlertSettings):
    """Save user alert preferences to a JSON file."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings.dict(), f, indent=2)

    return {"status": "success", "saved": settings}


@router.get("/settings/alerts")
def load_alert_settings():
    """Return saved alert settings."""
    if not os.path.exists(SETTINGS_FILE):
        return {
            "critical": {"email": True, "sms": False, "push": True, "sound": True},
            "warning": {"email": False, "sms": False, "push": True, "sound": True},
            "info": {"email": False, "sms": False, "push": True, "sound": False},
        }

    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)
