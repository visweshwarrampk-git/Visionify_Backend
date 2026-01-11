import os
import sys
import time
import urllib.request
from pathlib import Path
import multiprocessing as mp

import cv2
import numpy as np
from ultralytics import YOLO
import requests

# --- PySide6 Imports ---
from PySide6.QtCore import QTimer, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QPushButton,
    QInputDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
)

# --- Local Imports ---
from stream_widget import StreamWidget, convert_cv_to_qimage
from email_worker import EmailWorker

# --- Configuration ---
MODEL_PATH = "best.pt"
API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Section logic
SECTION_OPTIONS = ["heavy_machinery", "finishing_electrical", "structural"]
SECTION_REQUIREMENTS = {
    "heavy_machinery": ["no_helmet", "no_gloves"],
    "finishing_electrical": ["no_gloves", "no_goggle"],
    "structural": ["no_helmet", "no_gloves", "no_goggle", "no_boots"],
}

# Shared memory
MP_MANAGER = None
SHARED_FRAMES = None
STOP_EVENT = None
ACTIVE_WORKERS = {}
ALERT_QUEUE = None
EMAIL_SENDER_THREAD = None


# --- MJPEG Reader Helper ---
def mjpeg_frame_generator(url):
    """Manual MJPEG stream reader"""
    print(f"[MJPEG] Starting reader for: {url}")
    stream = urllib.request.urlopen(url)
    bytes_data = b""
    while True:
        bytes_data += stream.read(1024)
        a = bytes_data.find(b"\xff\xd8")
        b = bytes_data.find(b"\xff\xd9")
        if a != -1 and b != -1:
            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


# --- YOLO Worker ---
class YoloDetectorWorker(mp.Process):
    VIOLATION_CLASSES = {
        7: "no_helmet",
        8: "no_goggle",
        9: "no_gloves",
        10: "no_boots",
    }

    COOLDOWN_SECONDS = 10.0

    def __init__(self, stream_id, source, shared_frames_dict, stop_event, alert_queue, section="structural"):
        super().__init__()
        self.stream_id = stream_id
        self.source = source
        self._shared_frames = shared_frames_dict
        self._stop_event = stop_event
        self._alert_queue = alert_queue
        self._last_called = {}
        self.section = section if section in SECTION_REQUIREMENTS else "structural"
        self.daemon = True

    def _post_alert_to_api(self, violation_name):
        """Post alert only if PPE relevant for section"""
        allowed = SECTION_REQUIREMENTS.get(self.section, [])
        if violation_name not in allowed:
            return

        now = time.time()
        key = f"{self.stream_id}:{violation_name}"
        if (now - self._last_called.get(key, 0)) < self.COOLDOWN_SECONDS:
            return

        url = f"{API_BASE}/alerts/{violation_name}"
        payload = {"stream_id": self.stream_id, "violation": violation_name}
        try:
            resp = requests.post(url, json=payload, timeout=3)
            if resp.status_code in (200, 201):
                print(f"[{self.stream_id}] ✅ Alert sent: {violation_name}")
            else:
                print(f"[{self.stream_id}] ⚠️ Alert failed: {resp.status_code}")
        except Exception as e:
            print(f"[{self.stream_id}] ❌ API error: {e}")
        finally:
            self._last_called[key] = now

    def _open_capture(self):
        """Open any supported source (file, webcam, RTSP, MJPEG)"""
        src = self.source
        if str(src).startswith("http") and src.endswith(".mjpg"):
            return "mjpeg", mjpeg_frame_generator(src)
        cap_source = int(src) if str(src).isdigit() else str(src)
        cap = cv2.VideoCapture(cap_source)
        time.sleep(0.5)
        if not cap.isOpened():
            print(f"[{self.stream_id}] ❌ Failed to open: {src}")
            return None, None
        print(f"[{self.stream_id}] ✅ Stream opened: {src}")
        return "cv2", cap

    def run(self):
        print(f"[{self.stream_id}] 🚀 Worker started for section: {self.section}")
        try:
            model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"[{self.stream_id}] ❌ Model load failed: {e}")
            return

        source_type, reader = self._open_capture()
        if reader is None:
            print(f"[{self.stream_id}] ❌ No valid stream source. Exiting worker.")
            return

        cap = reader if source_type == "cv2" else None
        frame_gen = reader if source_type == "mjpeg" else None

        retry_count = 0
        while not self._stop_event.is_set():
            frame = None
            ret = False

            if source_type == "mjpeg":
                try:
                    frame = next(frame_gen)
                    ret = True
                except Exception:
                    frame, ret = None, False
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                retry_count += 1
                print(f"[{self.stream_id}] ⚠️ Frame read failed ({retry_count})")
                if retry_count > 20:
                    print(f"[{self.stream_id}] ❌ Too many failures, stopping worker.")
                    break
                time.sleep(0.2)
                continue
            retry_count = 0

            # Run YOLO detection
            try:
                results = model(frame, verbose=False)
                annotated = results[0].plot()
                fps = model.predictor.speed["fps"] if hasattr(model, "predictor") else 0
            except Exception as e:
                print(f"[{self.stream_id}] ⚠️ YOLO error: {e}")
                annotated = frame.copy()
                fps = 0

            detected_classes = set()
            try:
                detected_classes = set(int(c) for c in results[0].boxes.cls.tolist())
            except Exception:
                pass

            for cls_id, violation in self.VIOLATION_CLASSES.items():
                if cls_id in detected_classes:
                    self._post_alert_to_api(violation)

            try:
                self._shared_frames[self.stream_id] = (annotated, fps)
            except Exception:
                pass

        if cap:
            cap.release()
        if self.stream_id in self._shared_frames:
            del self._shared_frames[self.stream_id]
        print(f"[{self.stream_id}] 🛑 Worker stopped.")


# --- GUI Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Multi-Stream Dashboard")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)
        bar = QHBoxLayout()

        self.add_btn = QPushButton("Add Stream")
        self.add_btn.clicked.connect(self._on_add_stream_clicked)
        self.stop_btn = QPushButton("Stop All Streams")
        self.stop_btn.clicked.connect(self._on_stop_all_clicked)

        bar.addWidget(self.add_btn)
        bar.addWidget(self.stop_btn)
        layout.addLayout(bar)

        self.stream_grid_widget = QWidget()
        self.stream_grid = QGridLayout(self.stream_grid_widget)
        layout.addWidget(self.stream_grid_widget)

        self.stream_widgets = {}
        self._start_gui_update_timer()

    def _start_email_worker(self):
        global EMAIL_SENDER_THREAD, ALERT_QUEUE, STOP_EVENT
        if EMAIL_SENDER_THREAD is None or not EMAIL_SENDER_THREAD.is_alive():
            EMAIL_SENDER_THREAD = EmailWorker(ALERT_QUEUE, STOP_EVENT)
            EMAIL_SENDER_THREAD.start()

    @Slot()
    def _on_add_stream_clicked(self):
        text, ok = QInputDialog.getText(self, "Add Stream", "Enter source (0, test.mp4, rtsp://...):")
        if not ok or not text.strip():
            return
        sources = [s.strip() for s in text.split(",") if s.strip()]

        sec, ok2 = QInputDialog.getItem(self, "Select Section", "Choose section:", SECTION_OPTIONS, 0, False)
        if not ok2:
            return
        section = sec

        for src in sources:
            if not src:
                continue

            final = src
            if not src.isdigit() and not (src.startswith("http") or src.startswith("rtsp")):
                p = Path(src)
                if p.is_file():
                    final = str(p)
                elif Path(os.getcwd(), src).is_file():
                    final = str(Path(os.getcwd(), src))
                else:
                    QMessageBox.critical(self, "Error", f"File not found: {src}")
                    continue

            try:
                requests.post(f"{API_BASE}/streams", json={"source": final, "section": section}, timeout=2)
            except Exception:
                pass

            self._launch_worker(final, section)

    def _launch_worker(self, source, section):
        global ACTIVE_WORKERS, SHARED_FRAMES, STOP_EVENT, ALERT_QUEUE
        sid = Path(source).name if not str(source).isdigit() else str(source)
        if sid in ACTIVE_WORKERS and ACTIVE_WORKERS[sid].is_alive():
            QMessageBox.warning(self, "Warning", f"Stream '{sid}' already running.")
            return

        worker = YoloDetectorWorker(sid, source, SHARED_FRAMES, STOP_EVENT, ALERT_QUEUE, section)
        worker.start()
        ACTIVE_WORKERS[sid] = worker

        widget = StreamWidget(sid)
        self.stream_widgets[sid] = widget
        r, c = divmod(len(self.stream_widgets) - 1, 2)
        self.stream_grid.addWidget(widget, r, c)

    @Slot()
    def _on_stop_all_clicked(self):
        global STOP_EVENT, ACTIVE_WORKERS, MP_MANAGER
        STOP_EVENT.set()
        for w in list(ACTIVE_WORKERS.values()):
            if w.is_alive():
                w.terminate()
        ACTIVE_WORKERS.clear()
        for w in list(self.stream_widgets.values()):
            w.deleteLater()
        self.stream_widgets.clear()
        if MP_MANAGER:
            try:
                MP_MANAGER.shutdown()
            except Exception:
                pass
        self._init_shared_resources()
        self._start_email_worker()
        QMessageBox.information(self, "Stopped", "✅ All streams stopped.")

    def _start_gui_update_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_display)
        self.timer.start(40)

    @Slot()
    def _update_display(self):
        global SHARED_FRAMES
        current = dict(SHARED_FRAMES)
        for sid, widget in list(self.stream_widgets.items()):
            if sid in current:
                frame, fps = current[sid]
                qimg = convert_cv_to_qimage(frame)
                widget.update_frame(qimg, fps)
            else:
                widget.show_stopped_state()

    def _init_shared_resources(self):
        global MP_MANAGER, SHARED_FRAMES, STOP_EVENT, ALERT_QUEUE
        MP_MANAGER = mp.Manager()
        SHARED_FRAMES = MP_MANAGER.dict()
        STOP_EVENT = mp.Event()
        ALERT_QUEUE = MP_MANAGER.Queue()
        print("✅ Shared resources initialized")

    def closeEvent(self, event: QCloseEvent):
        global STOP_EVENT, ACTIVE_WORKERS, EMAIL_SENDER_THREAD, MP_MANAGER
        STOP_EVENT.set()
        for w in list(ACTIVE_WORKERS.values()):
            if w.is_alive():
                w.terminate()
        if EMAIL_SENDER_THREAD and EMAIL_SENDER_THREAD.is_alive():
            EMAIL_SENDER_THREAD.join(timeout=1)
        if MP_MANAGER:
            MP_MANAGER.shutdown()
        print("👋 Closed successfully.")
        event.accept()


if __name__ == "__main__":
    mp.freeze_support()
    app = QApplication(sys.argv)
    win = MainWindow()
    win._init_shared_resources()
    win._start_email_worker()
    win.show()
    sys.exit(app.exec())
