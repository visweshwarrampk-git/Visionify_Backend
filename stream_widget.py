# stream_widget.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QSize
import numpy as np
import cv2


# --- Helper Function for CV2 to QImage Conversion ---
def convert_cv_to_qimage(cv_img: np.ndarray) -> QImage:
    """Converts a numpy array (OpenCV BGR image) to QImage (RGB)."""
    # Check for empty frame (important for safety)
    if cv_img is None or cv_img.size == 0:
        return QImage()

    h, w, ch = cv_img.shape
    # OpenCV uses BGR, Qt uses RGB — conversion required
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    bytes_per_line = ch * w

    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


# --- Stream Display Widget ---
class StreamWidget(QWidget):
    """Widget to display live video frame and FPS."""

    def __init__(self, stream_id: str):
        super().__init__()
        self.stream_id = stream_id

        # --- Layout ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        # --- Video Frame Label ---
        self.frame_label = QLabel("Loading Stream...")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.frame_label)

        # --- Info Label ---
        self.info_label = QLabel(f"Stream ID: {stream_id} | FPS: --")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.info_label)

    def update_frame(self, frame_qimage: QImage, fps: float):
        """Receives QImage and FPS and updates the display."""
        if not frame_qimage.isNull():
            pixmap = QPixmap.fromImage(frame_qimage)
            display_size = QSize(640, 360)
            scaled_pixmap = pixmap.scaled(
                display_size, Qt.AspectRatioMode.KeepAspectRatio
            )
            self.frame_label.setPixmap(scaled_pixmap)
            self.info_label.setText(f"Stream ID: {self.stream_id} | FPS: {fps:.2f}")

    def show_stopped_state(self):
        """Show message when stream stops or is unavailable."""
        self.frame_label.setText("Stream Stopped (or Not Found)")
        self.info_label.setText(f"Stream ID: {self.stream_id} | Status: Stopped")
