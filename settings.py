from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class AlertPreferences(BaseModel):
    critical: dict
    warning: dict
    info: dict

class ContactInfo(BaseModel):
    email: str
    phone: str

# Temporary in-memory DB (later you can use SQLite/Postgres)
db_settings = {
    "preferences": {
        "critical": {"email": False, "sms": False, "push": False, "sound": False},
        "warning": {"email": False, "sms": False, "push": False, "sound": False},
        "info": {"email": False, "sms": False, "push": False, "sound": False},
    },
    "contact": { "email": "", "phone": "" }
}

@router.get("/settings/alerts")
def get_alert_settings():
    return db_settings["preferences"]

@router.post("/settings/alerts")
def update_alert_settings(prefs: AlertPreferences):
    db_settings["preferences"] = prefs.dict()
    return {"status": "ok"}

@router.post("/settings/contact")
def update_contact_info(info: ContactInfo):
    db_settings["contact"] = info.dict()
    return {"status": "ok"}
