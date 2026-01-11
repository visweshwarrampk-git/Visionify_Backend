# email_worker.py

import os
import cv2
import json
import time
import smtplib
import datetime
import threading

from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders


# ============================================================
# EMAIL CONFIGURATION
# ============================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "visweshwarrampk@gmail.com"
SENDER_PASSWORD = "rdpv gcta uqpb kknw"      # Gmail App Password
RECEIVER_EMAIL = "visweshwarrampk@gmail.com"

SETTINGS_FILE = "alert_settings.json"


# ============================================================
# GLOBAL THROTTLING
# ============================================================
LAST_ALERT_TIME = {}
ALERT_THROTTLE_SECONDS = 30  # seconds


# ============================================================
# SETTINGS LOADER
# ============================================================
def load_settings():
    """Load alert preference settings."""
    if not os.path.exists(SETTINGS_FILE):
        print("⚠️ No settings file found. Using defaults (all enabled).")
        return {
            "critical": {"email": True, "sms": False, "push": False, "sound": False},
            "warning": {"email": True, "sms": False, "push": False, "sound": False},
            "info": {"email": True, "sms": False, "push": False, "sound": False},
        }

    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


def is_email_enabled(violation_name: str):
    """Return whether email should be sent depending on category."""
    settings = load_settings()

    # Critical alerts (including dwell and group_monitoring)
    if violation_name in ["no_helmet", "fall", "restricted_area", "fire", "dwell", "group_monitoring"]:
        category = "critical"

    # Warning alerts
    elif violation_name in ["no_goggle", "no_gloves", "no_boots", "mobile_phone", "smoke"]:
        category = "warning"

    else:
        category = "info"

    return settings.get(category, {}).get("email", False)


# ============================================================
# MAIN EMAIL WORKER
# ============================================================
class EmailWorker(threading.Thread):
    """Background thread for processing alert emails."""

    def __init__(self, alert_queue, stop_event):
        super().__init__()
        self.alert_queue = alert_queue
        self.stop_event = stop_event
        self.daemon = True

        print("📩 EmailWorker initialized. Monitoring started...")

    # --------------------------------------------------------
    # THREAD LOOP
    # --------------------------------------------------------
    def run(self):
        if self.stop_event is None:
            print("❌ ERROR: stop_event is None. Worker cannot start.")
            return

        while not self.stop_event.is_set():
            try:
                stream_id, violation_name, frame_np = self.alert_queue.get(timeout=1)
                key = f"{stream_id}_{violation_name}"

                now = time.time()
                last = LAST_ALERT_TIME.get(key, 0)

                # Throttle alerts
                if (now - last) < ALERT_THROTTLE_SECONDS:
                    print(
                        f"[EmailWorker] ⏳ Throttled '{violation_name}' on '{stream_id}'. "
                        f"{now - last:.1f}s since last alert."
                    )
                    continue

                LAST_ALERT_TIME[key] = now
                print(f"[EmailWorker] ⚠️ Received alert: {violation_name} on {stream_id}")

                # Skip if email disabled
                if not is_email_enabled(violation_name):
                    print(f"🚫 Email disabled for '{violation_name}'. Skipping.")
                    continue

                # --------------------------------------------------------
                # BUILD EMAIL CONTENT
                # --------------------------------------------------------
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # --------------------
                # CRITICAL ALERTS
                # --------------------
                if violation_name == "fall":
                    subject = f"⚠️ CRITICAL ALERT: FALL INCIDENT (Stream {stream_id})"
                    body = (
                        f"A FALL INCIDENT occurred in stream '{stream_id}' at {timestamp}.\n\n"
                        f"Please review the attached snapshot immediately.\n\n"
                        f"— Safety Monitoring System"
                    )

                elif violation_name == "restricted_area":
                    subject = f"⚠️ RESTRICTED AREA ALERT (Stream {stream_id})"
                    body = (
                        f"A person ENTERED the RESTRICTED AREA in stream '{stream_id}' at {timestamp}.\n\n"
                        f"This is a high-risk security & safety violation.\n"
                        f"Snapshot attached.\n\n"
                        f"— Safety Monitoring System"
                    )

                elif violation_name == "fire":
                    subject = f"🔥 CRITICAL FIRE ALERT (Stream {stream_id})"
                    body = (
                        f"🔥 **FIRE DETECTED** in stream '{stream_id}' at {timestamp}.\n\n"
                        f"This is an emergency situation. Immediate action is required.\n"
                        f"Snapshot attached.\n\n"
                        f"— Fire & Safety Monitoring System"
                    )

                # NEW: DWELL TIME ALERT
                elif violation_name == "dwell":
                    subject = f"⏱️ CRITICAL: Excessive Dwell Time Alert (Stream {stream_id})"
                    body = (
                        f"⏱️ **EXCESSIVE DWELL TIME DETECTED** in stream '{stream_id}' at {timestamp}.\n\n"
                        f"A person has remained stationary for more than 4 seconds, indicating potential:\n"
                        f"- Safety hazard\n"
                        f"- Unauthorized loitering\n"
                        f"- Medical emergency\n\n"
                        f"Immediate investigation required. Snapshot attached.\n\n"
                        f"— Safety Monitoring System"
                    )
                # NEW: GROUP MONITORING ALERT
                elif violation_name == "group_monitoring":
                    subject = f"👥 CRITICAL: Multiple Persons Alert (Stream {stream_id})"
                    body = (
                        f"👥 **MULTIPLE PERSONS DETECTED** in restricted area in stream '{stream_id}' at {timestamp}.\n\n"
                        f"More number of persons detected in a area.\n\n"
                        f"Only ONE person is allowed at a time in this zone.\n"
                        f"Current detection: Multiple persons present.\n\n"
                        f"This is a critical safety and security violation.\n"
                        f"Snapshot attached.\n\n"
                        f"— Safety Monitoring System"
                    )

                # --------------------
                # WARNING ALERTS
                # --------------------
                elif violation_name == "smoke":
                    subject = f"🌫 WARNING: Smoke Detected (Stream {stream_id})"
                    body = (
                        f"Smoke was detected in stream '{stream_id}' at {timestamp}.\n\n"
                        f"Please investigate the source. Snapshot attached.\n\n"
                        f"— Safety Monitoring System"
                    )

                elif violation_name == "mobile_phone":
                    subject = f"⚠️ WARNING: Unauthorized Mobile Phone Use (Stream {stream_id})"
                    body = (
                        f"Unauthorized MOBILE PHONE usage detected in stream '{stream_id}' at {timestamp}.\n\n"
                        f"This zone prohibits mobile devices due to safety rules.\n"
                        f"Snapshot attached.\n\n"
                        f"— Safety Monitoring System"
                    )

                # --------------------
                # PPE WARNINGS
                # --------------------
                elif violation_name.startswith("no_"):
                    clean_name = violation_name.replace("no_", "").replace("_", " ").capitalize()
                    subject = f"⚠️ PPE Violation: {clean_name} Missing (Stream {stream_id})"
                    body = (
                        f"A person without **{clean_name.lower()}** detected in stream '{stream_id}' at {timestamp}.\n\n"
                        f"Snapshot attached.\n\n"
                        f"— PPE Monitoring System"
                    )

                else:
                    # fallback
                    subject = f"⚠️ Alert: {violation_name} (Stream {stream_id})"
                    body = f"Alert '{violation_name}' detected at {timestamp}.\nSnapshot attached."

                # Send email
                self._send_email(subject, body, RECEIVER_EMAIL, frame_np)

            except Exception:
                pass  # ignore timeout

        print("📨 EmailWorker stopped gracefully.")

    # --------------------------------------------------------
    # EMAIL SENDER
    # --------------------------------------------------------
    def _send_email(self, subject, body, to_email, frame=None):
        try:
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = SENDER_EMAIL
            msg["To"] = to_email
            msg.attach(MIMEText(body, "plain"))

            # Attach snapshot
            if frame is not None:
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    attachment = MIMEBase("image", "jpeg")
                    attachment.set_payload(buffer.tobytes())
                    encoders.encode_base64(attachment)
                    attachment.add_header(
                        "Content-Disposition", "attachment", filename="snapshot.jpg"
                    )
                    msg.attach(attachment)

            # Send email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, [to_email], msg.as_string())

            print(f"✅ Email sent → {subject}")

        except Exception as e:
            print(f"❌ Email sending failed: {e}")