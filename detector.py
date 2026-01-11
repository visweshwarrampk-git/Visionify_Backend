# detector.py (Unified PPE + Fall + Mobile + Restricted Area + Fire/Smoke + Dwell Time + Group Monitoring)

import time
import os
import cv2
import requests
import numpy as np
import multiprocessing as mp
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict

# -------------------------------
# MODEL PATHS
# -------------------------------
MODEL_PATH = "models/PPE_yolov8.pt"
POSE_MODEL_PATH = "models/Fall_yolo11s_pose.pt"
MOBILE_MODEL_PATH = "models/mobile_yolov8.pt"
PERSON_MODEL_PATH = "models/yolov8n_Person.pt"
FIRE_MODEL_PATH = "models/fire_smoke_best.pt"
DWELL_MODEL_PATH = "models/yolov8m_Prolongated_Time.pt"
RESTRICTED_AREA_PERSON_MODEL_PATH = "models/Restricted_Area_Person.pt"
GROUP_MONITORING_MODEL_PATH = "models/Group_Monitoring.pt"  # ✅ NEW

CELL_PHONE_CLASS_INDEX = 67

API_BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")

Y_DIFF_THRESHOLD = -50


# ============================================================
# DWELL TIME TRACKER CLASS
# ============================================================
class DwellTimeTracker:
    """
    Track persons and their dwell time without assigning IDs
    Uses centroid tracking for simplicity
    """
    def __init__(self, max_disappeared=30, dwell_threshold=4.0):
        """
        Args:
            max_disappeared: Max frames a person can disappear before being removed
            dwell_threshold: Dwell time threshold in seconds (default 4.0)
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.dwell_threshold = dwell_threshold
    
    def register(self, centroid, bbox):
        """Register a new person"""
        current_time = time.time()
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'first_seen': current_time,
            'last_seen': current_time,
            'alerted': False
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove a person from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """
        Update tracked persons with new detections
        
        Args:
            detections: List of (bbox, centroid) tuples
                       bbox = (x1, y1, x2, y2)
                       centroid = (cx, cy)
        
        Returns:
            OrderedDict of tracked objects with dwell time info
        """
        current_time = time.time()
        
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Extract centroids and bboxes from detections
        input_centroids = np.array([d[1] for d in detections])
        input_bboxes = [d[0] for d in detections]
        
        # If no objects tracked yet, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i])
        
        # Otherwise, match existing objects to new detections
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid]['centroid'] for oid in object_ids])
            
            # Compute distance between existing and new centroids
            D = dist.cdist(object_centroids, input_centroids)
            
            # Find minimum distance for each existing object
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match existing objects to new detections
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Check if distance is reasonable (< 80 pixels)
                if D[row, col] < 80:
                    object_id = object_ids[row]
                    self.objects[object_id]['centroid'] = input_centroids[col]
                    self.objects[object_id]['bbox'] = input_bboxes[col]
                    self.objects[object_id]['last_seen'] = current_time
                    self.disappeared[object_id] = 0
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Mark unmatched existing objects as disappeared
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new detections
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])
        
        return self.objects
    
    def get_dwell_times(self):
        """
        Get current dwell time for each tracked object
        
        Returns:
            Dict of {object_id: dwell_time_seconds}
        """
        current_time = time.time()
        dwell_times = {}
        
        for object_id, obj_data in self.objects.items():
            dwell_time = current_time - obj_data['first_seen']
            dwell_times[object_id] = dwell_time
        
        return dwell_times
    
    def get_violations(self):
        """
        Get list of persons who exceeded dwell threshold
        
        Returns:
            List of object_ids that exceeded threshold and haven't been alerted
        """
        violations = []
        dwell_times = self.get_dwell_times()
        
        for object_id, dwell_time in dwell_times.items():
            if dwell_time >= self.dwell_threshold and not self.objects[object_id]['alerted']:
                violations.append(object_id)
                self.objects[object_id]['alerted'] = True
        
        return violations


# ============================================================
# YOLO DETECTOR WORKER
# ============================================================
class YoloDetectorWorker(mp.Process):

    # PPE violation from custom model
    VIOLATION_CLASSES = {
        7: "no_helmet",
        8: "no_goggle",
        9: "no_gloves",
        10: "no_boots",
    }

    MOBILE_VIOLATION = "mobile_phone"
    RESTRICTED_AREA_VIOLATION = "restricted_area"
    GROUP_MONITORING_VIOLATION = "group_monitoring"  # ✅ NEW
    COOLDOWN_SECONDS = 10.0

    def __init__(
        self, stream_id, source, shared_frames_dict, stop_event,
        alert_queue, model_path=None, zone_rules=None, zone_id=None
    ):
        super().__init__()

        self.stream_id = stream_id
        self.source = source
        self._shared_frames = shared_frames_dict
        self._stop_event = stop_event
        self._alert_queue = alert_queue

        self.model_path = model_path or MODEL_PATH
        self._last_called = {}

        self.zone_id = zone_id if zone_id else "ZONE A"
        self.zone_rules = zone_rules.get(self.zone_id, {})
        print(f"[{self.stream_id} | Zone {self.zone_id}] Loaded rules: {self.zone_rules}")

        # Dwell time tracker
        self.dwell_tracker = DwellTimeTracker(
            max_disappeared=30,
            dwell_threshold=4.0
        )

        self.daemon = True

    # ------------------------------------------------
    def _call_api_alert(self, violation_name):

        # Map violations to zone rule keys
        if violation_name == "restricted_area":
            rule_key = "Restricted Area"
        elif violation_name == "dwell":
            rule_key = "Dwell"
        elif violation_name == "group_monitoring":  # ✅ NEW
            rule_key = "Group_Monitoring"
        else:
            rule_key = violation_name

        # Zone rule filtering
        if self.zone_rules.get(rule_key) is not True:
            return

        # Cooldown
        key = f"{self.stream_id}:{violation_name}"
        now = time.time()
        last = self._last_called.get(key, 0)
        if (now - last) < self.COOLDOWN_SECONDS:
            return

        url = f"{API_BASE}/alerts/{violation_name}"
        payload = {"stream_id": self.stream_id, "violation": violation_name}

        try:
            requests.post(url, json=payload, timeout=3)
        except:
            pass
        finally:
            self._last_called[key] = now

    # ------------------------------------------------
    def _check_fall_by_pose(self, keypoints, frame_height):

        try:
            nose = keypoints[0]
            lhip = keypoints[11]
            rhip = keypoints[12]

            if nose[2] < 0.5 or lhip[2] < 0.5 or rhip[2] < 0.5:
                return False

            avg_hip_y = (lhip[1] + rhip[1]) / 2
            vertical_diff = nose[1] - avg_hip_y

            return vertical_diff > Y_DIFF_THRESHOLD

        except IndexError:
            return False

    # ------------------------------------------------
    def process_dwell_time(self, frame, dwell_model):
        """
        Process dwell time detection for persons
        
        Args:
            frame: OpenCV frame
            dwell_model: YOLO model for person detection
        
        Returns:
            frame: Frame with dwell time bounding boxes drawn
        """
        if dwell_model is None:
            return frame
        
        # Check if dwell detection enabled for this zone
        if not self.zone_rules.get("Dwell", False):
            return frame
        
        try:
            # Run person detection
            results = dwell_model(frame, verbose=False, conf=0.5)
            
            # Extract person detections
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Get class (should be 'person' = class 0 in COCO)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Only process person detections with confidence > 0.5
                    if cls == 0 and conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate centroid
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        centroid = (cx, cy)
                        
                        detections.append((bbox, centroid))
            
            # Update tracker with new detections
            tracked_objects = self.dwell_tracker.update(detections)
            
            # Get dwell times
            dwell_times = self.dwell_tracker.get_dwell_times()
            
            # Get violations (people who exceeded threshold)
            violations = self.dwell_tracker.get_violations()
            
            # Trigger alerts for violations
            for object_id in violations:
                print(f"🚨 DWELL TIME VIOLATION: Person {object_id} exceeded {self.dwell_tracker.dwell_threshold}s in stream {self.stream_id}")
                
                # Call API to save to database and broadcast to frontend
                self._call_api_alert("dwell")
                
                # Send email alert
                self._alert_queue.put((self.stream_id, "dwell", frame.copy()))
            
            # Draw bounding boxes
            for object_id, obj_data in tracked_objects.items():
                bbox = obj_data['bbox']
                dwell_time = dwell_times[object_id]
                
                x1, y1, x2, y2 = bbox
                
                # Color: Green if < threshold, Red if >= threshold
                if dwell_time < self.dwell_tracker.dwell_threshold:
                    color = (0, 255, 0)  # Green
                    status = "OK"
                else:
                    color = (0, 0, 255)  # Red
                    status = "ALERT"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw dwell time text
                text = f"{status}: {dwell_time:.1f}s"
                
                # Background for text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0], y1),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            return frame
        
        except Exception as e:
            print(f"❌ Error in dwell time processing: {e}")
            import traceback
            traceback.print_exc()
            return frame

    # ------------------------------------------------
    def process_restricted_area_person(self, frame, restricted_area_model):
        """
        Process restricted area person detection using dedicated model
        
        Args:
            frame: OpenCV frame
            restricted_area_model: YOLO model trained to detect persons in restricted areas
        
        Returns:
            frame: Frame with restricted area violations drawn
        """
        if restricted_area_model is None:
            return frame
        
        # Check if restricted area detection enabled for this zone
        if not self.zone_rules.get("Restricted Area", False):
            return frame
        
        try:
            # Run restricted area person detection
            results = restricted_area_model(frame, verbose=False, conf=0.5)
            
            violation_detected = False
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Get class (class 0 = person in restricted area)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Process person in restricted area detections
                    if cls == 0 and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Draw RED bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Draw warning text
                        cv2.putText(
                            frame,
                            "RESTRICTED AREA VIOLATION!",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )
                        
                        # Draw confidence
                        cv2.putText(
                            frame,
                            f"Conf: {conf:.2f}",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                        
                        violation_detected = True
            
            # Trigger alert if violation detected
            if violation_detected:
                print(f"🚨 RESTRICTED AREA VIOLATION: Person detected in restricted area - Stream {self.stream_id}")
                
                # Call API to save to database
                self._call_api_alert("restricted_area")
                
                # Send email alert
                self._alert_queue.put((self.stream_id, "restricted_area", frame.copy()))
            
            return frame
        
        except Exception as e:
            print(f"❌ Error in restricted area person processing: {e}")
            import traceback
            traceback.print_exc()
            return frame

    # ------------------------------------------------
    # ✅ NEW: GROUP MONITORING DETECTION
    # ------------------------------------------------
    def process_group_monitoring(self, frame, group_monitoring_model):
        """
        Process group monitoring detection
        
        Logic:
        - 1 person detected → BLUE box (Safe)
        - 2+ persons detected → RED boxes (VIOLATION + Alert)
        
        Args:
            frame: OpenCV frame
            group_monitoring_model: YOLO model for group detection
        
        Returns:
            frame: Frame with group monitoring bounding boxes
        """
        if group_monitoring_model is None:
            return frame
        
        # Check if group monitoring enabled for this zone
        if not self.zone_rules.get("Group_Monitoring", False):
            return frame
        
        try:
            # Run group monitoring detection
            results = group_monitoring_model(frame, verbose=False, conf=0.5)
            
            # Collect all person detections
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Class 0 = person
                    if cls == 0 and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        person_boxes.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf
                        })
            
            # Count persons
            person_count = len(person_boxes)
            
            # Determine if violation (2+ persons)
            is_violation = person_count >= 2
            
            # Choose color based on count
            if person_count == 0:
                # No persons - no boxes
                pass
            elif person_count == 1:
                # 1 person - BLUE box (Safe)
                color = (255, 0, 0)  # Blue in BGR
                label_text = "Safe: 1 person"
                label_color = color
            else:
                # 2+ persons - RED boxes (VIOLATION)
                color = (0, 0, 255)  # Red in BGR
                label_text = f"VIOLATION: {person_count} persons!"
                label_color = color
            
            # Draw bounding boxes
            for person in person_boxes:
                x1, y1, x2, y2 = person['bbox']
                conf = person['conf']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw confidence
                cv2.putText(
                    frame,
                    f"Person: {conf:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Draw status label at top
            if person_count > 0:
                # Background for text
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.rectangle(
                    frame,
                    (10, 10),
                    (20 + text_size[0], 40 + text_size[1]),
                    label_color,
                    -1
                )
                
                # Text
                cv2.putText(
                    frame,
                    label_text,
                    (15, 35 + text_size[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3
                )
            
            # Trigger alert if violation (2+ persons)
            if is_violation:
                print(f"👥 GROUP MONITORING VIOLATION: {person_count} persons detected - Stream {self.stream_id}")
                
                # Call API to save to database
                self._call_api_alert("group_monitoring")
                
                # Send email alert
                self._alert_queue.put((self.stream_id, "group_monitoring", frame.copy()))
            
            return frame
        
        except Exception as e:
            print(f"❌ Error in group monitoring processing: {e}")
            import traceback
            traceback.print_exc()
            return frame

    # ------------------------------------------------
    def run(self):

        # Load Models
        try:
            ppe_model = YOLO(self.model_path)
        except Exception as e:
            print(f"[{self.stream_id}] PPE model error: {e}")
            return

        try:
            pose_model = YOLO(POSE_MODEL_PATH)
        except Exception as e:
            print(f"[{self.stream_id}] Pose model error: {e}")
            return

        try:
            mobile_model = YOLO(MOBILE_MODEL_PATH)
        except:
            mobile_model = None

        try:
            fire_model = YOLO(FIRE_MODEL_PATH)
            print(f"[{self.stream_id}] Fire/Smoke model loaded.")
        except Exception as e:
            print(f"[{self.stream_id}] Fire/Smoke model error: {e}")
            fire_model = None

        try:
            dwell_model = YOLO(DWELL_MODEL_PATH)
            print(f"[{self.stream_id}] ✅ Dwell Time model loaded: {DWELL_MODEL_PATH}")
        except Exception as e:
            print(f"[{self.stream_id}] ⚠️ Dwell Time model error: {e}")
            dwell_model = None

        try:
            restricted_area_person_model = YOLO(RESTRICTED_AREA_PERSON_MODEL_PATH)
            print(f"[{self.stream_id}] ✅ Restricted Area Person model loaded: {RESTRICTED_AREA_PERSON_MODEL_PATH}")
        except Exception as e:
            print(f"[{self.stream_id}] ⚠️ Restricted Area Person model error: {e}")
            restricted_area_person_model = None

        # ✅ NEW: Load Group Monitoring Model
        try:
            group_monitoring_model = YOLO(GROUP_MONITORING_MODEL_PATH)
            print(f"[{self.stream_id}] ✅ Group Monitoring model loaded: {GROUP_MONITORING_MODEL_PATH}")
        except Exception as e:
            print(f"[{self.stream_id}] ⚠️ Group Monitoring model error: {e}")
            group_monitoring_model = None

        # Video Source
        cap_src = int(self.source) if str(self.source).isdigit() else str(self.source)
        cap = cv2.VideoCapture(cap_src)

        if not cap.isOpened():
            print(f"[{self.stream_id}] Cannot open source {self.source}")
            return

        print(f"[{self.stream_id}] Detector started — PPE + Pose + Mobile + Fire/Smoke + Dwell Time + Restricted Area Person + Group Monitoring loaded.")

        # ---------------------------------------------------------
        # LOOP
        # ---------------------------------------------------------
        while not self._stop_event.is_set():

            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()
            frame_height, frame_width, _ = frame.shape

            # ===================================================
            # 1) PPE MODEL
            # ===================================================
            ppe_results = ppe_model(frame, verbose=False, conf=0.30)

            for result in ppe_results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name = result.names.get(cls, f"Class {cls}")

                    if cls in self.VIOLATION_CLASSES:
                        violation = self.VIOLATION_CLASSES[cls]

                        if self.zone_rules.get(violation) is True:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(
                                annotated, f"{name} {conf:.2f}",
                                (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                            )

                        self._call_api_alert(violation)

                    else:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            annotated, f"{name} {conf:.2f}",
                            (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                        )

            # ===================================================
            # 2) FALL DETECTION
            # ===================================================
            pose_results = pose_model(frame, verbose=False, conf=0.50)

            for result in pose_results:
                if result.keypoints is None:
                    continue

                for i, kp in enumerate(result.keypoints.data):
                    keypoints_np = kp.cpu().numpy()

                    if self._check_fall_by_pose(keypoints_np, frame_height):

                        if self.zone_rules.get("fall") is True:
                            box = result.boxes[i]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            cv2.putText(
                                annotated, "FALL DETECTED",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3
                            )

                            self._call_api_alert("fall")

            # ===================================================
            # 3) MOBILE PHONE DETECTION
            # ===================================================
            if mobile_model:
                mobile_results = mobile_model(
                    frame, verbose=False, classes=[CELL_PHONE_CLASS_INDEX], conf=0.50
                )

                for result in mobile_results:
                    if result.boxes is None:
                        continue

                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        if self.zone_rules.get("mobile_phone") is True:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                            cv2.putText(
                                annotated, f"MOBILE {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                            )

                        self._call_api_alert(self.MOBILE_VIOLATION)

            # ===================================================
            # 4) FIRE & SMOKE DETECTION
            # ===================================================
            if fire_model:
                try:
                    fire_results = fire_model(frame, verbose=False, conf=0.35)
                    for result in fire_results:
                        if result.boxes is None:
                            continue

                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if cls == 0:
                                violation_name = "fire"
                                if self.zone_rules.get("fire") is True:
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(
                                        annotated, f"FIRE {conf:.2f}",
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                                    )
                                self._call_api_alert(violation_name)

                            elif cls == 1:
                                violation_name = "smoke"
                                if self.zone_rules.get("smoke") is True:
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 200), 3)
                                    cv2.putText(
                                        annotated, f"SMOKE {conf:.2f}",
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2
                                    )
                                self._call_api_alert(violation_name)

                except Exception as e:
                    print(f"[{self.stream_id}] Fire model inference error: {e}")

            # ===================================================
            # 5) RESTRICTED AREA PERSON DETECTION
            # ===================================================
            annotated = self.process_restricted_area_person(annotated, restricted_area_person_model)

            # ===================================================
            # 6) GROUP MONITORING DETECTION (NEW)
            # ===================================================
            annotated = self.process_group_monitoring(annotated, group_monitoring_model)

            # ===================================================
            # 7) DWELL TIME DETECTION
            # ===================================================
            annotated = self.process_dwell_time(annotated, dwell_model)

            # --------------------------------------------------
            # Update shared frame
            # --------------------------------------------------
            try:
                self._shared_frames[self.stream_id] = (annotated, 0)
            except:
                pass

        # Cleanup
        cap.release()

        if self.stream_id in self._shared_frames:
            del self._shared_frames[self.stream_id]

        print(f"[{self.stream_id}] Detector stopped.")