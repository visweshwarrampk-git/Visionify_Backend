# frontend_GUI.py

import sys
import io
import time
import json
import asyncio
import threading
import requests
import websockets

from PIL import Image

from PySide6.QtCore import Qt, QTimer, QCoreApplication
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QGridLayout, QInputDialog, QMessageBox, QHBoxLayout
)

# ---------------------------------------------------------
# SERVER ENDPOINTS
# ---------------------------------------------------------
API_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/alerts"

ZONE_OPTIONS = ["ZONE A", "ZONE B", "ZONE C", "ZONE D", "Break Room"]


# ---------------------------------------------------------
# IMAGE DECODER
# ---------------------------------------------------------
def convert_bytes_to_qimage(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        data = img.tobytes("raw", "RGB")
        return QImage(data, img.width, img.height, QImage.Format.Format_RGB888)
    except Exception as e:
        print("Image conversion error:", e)
        return QImage()


# =========================================================
# STREAM WIDGET (Each camera box)
# =========================================================
class StreamWidget(QWidget):
    def __init__(self, stream_id, stop_callback):
        super().__init__()

        self.stream_id = stream_id
        self.stop_callback = stop_callback

        self.current_alert = ""
        self.alert_expire_time = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Video frame
        self.frame_label = QLabel("Loading Stream…")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setFixedSize(640, 360)
        layout.addWidget(self.frame_label)

        # Alert banner
        self.alert_label = QLabel("")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setStyleSheet(
            "background:#550000;color:white;padding:5px;font-size:16px;"
        )
        self.alert_label.hide()
        layout.addWidget(self.alert_label)

        # Info row
        info_row = QHBoxLayout()
        info_row.addWidget(QLabel(f"Stream ID: {stream_id}"))

        stop_btn = QPushButton("Stop Stream")
        stop_btn.clicked.connect(self.on_stop)
        info_row.addWidget(stop_btn)

        layout.addLayout(info_row)

    # -----------------------------------------------------
    def update_frame(self, qimg):
        if not qimg.isNull():
            pix = QPixmap.fromImage(qimg).scaled(640, 360, Qt.KeepAspectRatio)
            self.frame_label.setPixmap(pix)

        # Auto-hide alerts after expiry
        if self.current_alert and time.time() > self.alert_expire_time:
            self.alert_label.hide()
            self.current_alert = ""

    # -----------------------------------------------------
    def show_alert(self, violation):
        # COLOR MAPPING
        colors = {
            "no_helmet": "#8A0303",
            "fall": "#8A008A",
            "no_goggle": "#9A4F00",
            "no_gloves": "#9A4F00",
            "no_boots": "#9A4F00",
            "mobile_phone": "#005577",
            "restricted_area": "#FF0000",
            "fire": "#FF4500",
            "smoke": "#AAAAAA",
            "dwell": "#FF6600",
            "group_monitoring": "#FF1493"  # ✅ NEW: Deep Pink for group monitoring
        }

        # readable label
        label = violation.upper().replace("_", " ")

        # PPE naming fix
        if violation.startswith("no_"):
            label = f"{label.replace('NO ', '')} MISSING"

        # Special naming
        if violation == "fire":
            label = "FIRE DETECTED"
        elif violation == "smoke":
            label = "SMOKE DETECTED"
        elif violation == "dwell":
            label = "EXCESSIVE DWELL TIME"
        elif violation == "group_monitoring":  # ✅ NEW
            label = "MULTIPLE PERSONS DETECTED"

        self.alert_label.setText(f"⚠ {label}")
        self.alert_label.setStyleSheet(
            f"background:{colors.get(violation, '#444')};"
            "color:white;padding:6px;font-size:16px;"
        )
        self.alert_label.show()

        self.current_alert = violation
        self.alert_expire_time = time.time() + 5

    # -----------------------------------------------------
    def on_stop(self):
        self.stop_callback(self.stream_id)
        self.frame_label.setText("Stream stopped.")
        self.alert_label.hide()


# =========================================================
# MAIN GUI WINDOW
# =========================================================
class HelmetAIFrontend(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PPE, Fall, Mobile, Fire, Smoke & Dwell Time Detection Dashboard")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top Bar
        top_row = QHBoxLayout()
        self.btn_add = QPushButton("Add Stream")
        self.btn_add.clicked.connect(self.add_stream)

        self.btn_stop_all = QPushButton("Stop All")
        self.btn_stop_all.clicked.connect(self.stop_all)

        top_row.addWidget(self.btn_add)
        top_row.addWidget(self.btn_stop_all)
        layout.addLayout(top_row)

        # Grid for streams
        self.stream_widgets = {}
        self.stream_grid = QGridLayout()
        layout.addLayout(self.stream_grid)

        # Timer for frame refresh
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_frames)
        self.timer.start(800)

        # Background threads
        threading.Thread(target=self.stream_sync_loop, daemon=True).start()
        threading.Thread(target=self.start_ws_listener, daemon=True).start()

    # ---------------------------------------------------------
    def add_stream(self):
        source, ok = QInputDialog.getText(
            self, "Add Stream", "Enter source (0 / file.mp4 / URL):"
        )
        if not ok or not source.strip():
            return

        zone, ok2 = QInputDialog.getItem(
            self,
            "Select Zone",
            "Choose Construction Zone:",
            ZONE_OPTIONS,
            0,
            editable=False
        )
        if not ok2:
            return

        payload = {"source": source.strip(), "zone_id": zone}

        try:
            r = requests.post(f"{API_URL}/streams", json=payload)
            if r.status_code == 200:
                sid = r.json()["stream_id"]
                self.add_widget(sid)
                QMessageBox.information(self, "Success", f"Stream {sid} started.")
            else:
                QMessageBox.critical(self, "Error", r.text)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def add_widget(self, sid):
        if sid not in self.stream_widgets:
            widget = StreamWidget(sid, self.stop_stream)
            self.stream_widgets[sid] = widget

            pos = len(self.stream_widgets) - 1
            self.stream_grid.addWidget(widget, pos // 2, pos % 2)

    # ---------------------------------------------------------
    def stop_stream(self, sid):
        try:
            requests.delete(f"{API_URL}/streams/{sid}")
            if sid in self.stream_widgets:
                self.stream_widgets[sid].deleteLater()
                del self.stream_widgets[sid]
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def stop_all(self):
        try:
            requests.post(f"{API_URL}/streams/stop_all")
            for w in self.stream_widgets.values():
                w.deleteLater()
            self.stream_widgets.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ---------------------------------------------------------
    def refresh_frames(self):
        for sid, widget in list(self.stream_widgets.items()):
            try:
                r = requests.get(f"{API_URL}/streams/{sid}/snapshot", timeout=2)
                if r.status_code == 200:
                    widget.update_frame(convert_bytes_to_qimage(r.content))
            except:
                pass

    # ---------------------------------------------------------
    def stream_sync_loop(self):
        while True:
            try:
                r = requests.get(f"{API_URL}/streams")
                active = r.json().get("streams", {})

                # Add missing widgets
                for sid in active:
                    if sid not in self.stream_widgets:
                        QCoreApplication.instance().callOnMainThread(
                            lambda sid=sid: self.add_widget(sid)
                        )

                # Remove widgets for stopped streams
                for sid in list(self.stream_widgets.keys()):
                    if sid not in active:
                        QCoreApplication.instance().callOnMainThread(
                            lambda sid=sid: (
                                self.stream_widgets[sid].deleteLater(),
                                self.stream_widgets.pop(sid, None)
                            )
                        )
            except:
                pass

            time.sleep(3)

    # ---------------------------------------------------------
    async def ws_handler(self):
        while True:
            try:
                async with websockets.connect(WS_URL, open_timeout=5) as ws:
                    print("WS Connected")
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)

                        sid = data["stream_id"]
                        violation = data["violation"]

                        if sid in self.stream_widgets:
                            QCoreApplication.instance().callOnMainThread(
                                lambda v=violation: self.stream_widgets[sid].show_alert(v)
                            )

            except Exception as e:
                print("WS error:", e)
                await asyncio.sleep(2)

    def start_ws_listener(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ws_handler())


# =========================================================
# MAIN ENTRY
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = HelmetAIFrontend()
    gui.show()
    sys.exit(app.exec())