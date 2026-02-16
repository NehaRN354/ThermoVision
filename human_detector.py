
"""
ThermoVision - Human Detection Module
Detects humans with proximity awareness and approach tracking
Only alerts when people are dangerously close or approaching rapidly
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import math


class HumanDetector:
    """
    Intelligent human detection with:
    - Distance estimation (near/medium/far)
    - Approach velocity tracking
    - Front/back orientation detection
    - Smart alerting (only close proximity)
    """

    def __init__(self,
                 confidence_threshold=0.5,
                 alert_distance_threshold=100,  # pixels - VERY close
                 warning_distance_threshold=200,  # pixels - close
                 approach_velocity_threshold=15):  # pixels/frame - fast approach
        """
        Initialize human detector

        Args:
            confidence_threshold: Minimum confidence for person detection
            alert_distance_threshold: Critical proximity (trigger alert)
            warning_distance_threshold: Warning proximity (caution)
            approach_velocity_threshold: Speed threshold for approach alerts
        """
        self.confidence_threshold = confidence_threshold
        self.alert_distance_threshold = alert_distance_threshold
        self.warning_distance_threshold = warning_distance_threshold
        self.approach_velocity_threshold = approach_velocity_threshold

        # Initialize YOLO model (YOLOv8 nano for speed)
        print("👥 Loading YOLO model for human detection...")
        try:
            self.model = YOLO('yolov8n.pt')  # Nano model for real-time
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠️ YOLO model download in progress: {e}")
            self.model = YOLO('yolov8n.pt')  # Will auto-download

        # Person class ID in COCO dataset
        self.person_class_id = 0

        # Tracking history for velocity calculation
        self.person_history = {}  # {person_id: deque of (x, y, w, h, timestamp)}
        self.next_person_id = 0

        # Alert cooldown management
        self.last_alert_time = {}  # {person_id: timestamp}
        self.alert_cooldown = 3.0  # seconds

        # Performance tracking
        self.inference_times = deque(maxlen=30)

        print("✅ Human detector initialized")
        print(f"   Alert distance: <{alert_distance_threshold}px (VERY close)")
        print(f"   Warning distance: <{warning_distance_threshold}px (close)")
        print(f"   Approach threshold: >{approach_velocity_threshold}px/frame")

    def detect(self, frame):
        """
        Detect humans with proximity and approach analysis

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict: {
                'persons': list of person detections,
                'critical_alerts': list of close proximity persons,
                'warnings': list of approaching persons,
                'total_persons': int,
                'inference_time': float
            }
        """
        start_time = time.time()

        # Run YOLO detection
        results = self.model(frame, verbose=False)

        # Extract person detections
        persons = []
        frame_height, frame_width = frame.shape[:2]

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Check if it's a person
                class_id = int(box.cls[0])
                if class_id != self.person_class_id:
                    continue

                # Check confidence
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

                # Estimate distance (inverse of box height)
                # Larger person in frame = closer
                distance_score = self._estimate_distance(h, frame_height)

                # Detect direction (front/back/side)
                orientation = self._detect_orientation(frame, x, y, w, h)

                # Calculate position zone
                zone = self._get_position_zone(x, y, w, h, frame_width, frame_height)

                person = {
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'distance_score': distance_score,
                    'distance_category': self._categorize_distance(distance_score),
                    'orientation': orientation,
                    'zone': zone,
                    'center': (x + w//2, y + h//2),
                    'timestamp': time.time()
                }

                persons.append(person)

        # Track persons and calculate velocities
        persons = self._track_and_calculate_velocity(persons)

        # Categorize by threat level
        critical_alerts = []
        warnings = []

        for person in persons:
            threat_level = self._assess_threat_level(person)
            person['threat_level'] = threat_level

            if threat_level == 'critical':
                critical_alerts.append(person)
            elif threat_level == 'warning':
                warnings.append(person)

        # Performance tracking
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        result = {
            'persons': persons,
            'critical_alerts': critical_alerts,
            'warnings': warnings,
            'total_persons': len(persons),
            'inference_time': inference_time
        }

        return result

    def _estimate_distance(self, person_height, frame_height):
        """
        Estimate distance based on person height in frame

        Args:
            person_height: Height of person bounding box (pixels)
            frame_height: Total frame height

        Returns:
            float: Distance score (0-1, where 1 = very close)
        """
        # Normalize by frame height
        height_ratio = person_height / frame_height

        # Convert to distance score
        # If person occupies >50% of frame height = very close (score ~1.0)
        # If person occupies <10% of frame height = far (score ~0.2)
        distance_score = min(height_ratio * 2, 1.0)

        return distance_score

    def _categorize_distance(self, distance_score):
        """
        Categorize distance into near/medium/far
        LESS SENSITIVE - only "near" when VERY close

        Args:
            distance_score: 0-1 distance score

        Returns:
            str: 'near', 'medium', or 'far'
        """
        # MUCH STRICTER - only alert when extremely close
        if distance_score > 0.6:  # Person occupies >60% frame height (INCREASED from 0.5)
            return 'near'
        elif distance_score > 0.35:  # INCREASED from 0.25
            return 'medium'
        else:
            return 'far'

    def _detect_orientation(self, frame, x, y, w, h):
        """
        Detect if person is facing towards/away/sideways
        Uses basic heuristics (can be improved with pose estimation)

        Args:
            frame: Input frame
            x, y, w, h: Bounding box

        Returns:
            str: 'front', 'back', 'side', or 'unknown'
        """
        # Extract person ROI
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            return 'unknown'

        # Simple heuristic: analyze vertical symmetry
        # Front faces tend to be more symmetric than backs

        # Split ROI vertically
        mid = w // 2
        left_half = roi[:, :mid]
        right_half = cv2.flip(roi[:, mid:], 1)  # Flip for comparison

        # Resize to match if odd width
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

        # Calculate similarity
        if left_half.shape == right_half.shape:
            diff = cv2.absdiff(left_half, right_half)
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)

            # Higher symmetry = likely front-facing
            if symmetry_score > 0.7:
                return 'front'
            elif symmetry_score < 0.5:
                return 'back'
            else:
                return 'side'

        return 'unknown'

    def _get_position_zone(self, x, y, w, h, frame_width, frame_height):
        """
        Determine which zone the person is in (left/center/right)

        Args:
            x, y, w, h: Bounding box
            frame_width, frame_height: Frame dimensions

        Returns:
            str: Position zone
        """
        center_x = x + w // 2
        center_y = y + h // 2

        # Horizontal zones
        if center_x < frame_width / 3:
            h_zone = 'left'
        elif center_x < 2 * frame_width / 3:
            h_zone = 'center'
        else:
            h_zone = 'right'

        # Vertical zones (for distance)
        if center_y > 2 * frame_height / 3:
            v_zone = 'near'
        elif center_y > frame_height / 3:
            v_zone = 'mid'
        else:
            v_zone = 'far'

        return f"{h_zone}-{v_zone}"

    def _track_and_calculate_velocity(self, persons):
        """
        Track persons across frames and calculate approach velocity

        Args:
            persons: List of detected persons

        Returns:
            List of persons with velocity information
        """
        current_time = time.time()

        # Match current detections with tracked persons
        matched_persons = []

        for person in persons:
            center = person['center']

            # Find closest tracked person (simple nearest neighbor)
            best_match_id = None
            best_distance = float('inf')

            for person_id, history in self.person_history.items():
                if len(history) == 0:
                    continue

                last_center = history[-1]['center']
                dist = math.sqrt(
                    (center[0] - last_center[0])**2 +
                    (center[1] - last_center[1])**2
                )

                # Match if within reasonable distance (same person)
                if dist < 100 and dist < best_distance:
                    best_distance = dist
                    best_match_id = person_id

            # Assign ID
            if best_match_id is not None:
                person['id'] = best_match_id
            else:
                person['id'] = self.next_person_id
                self.next_person_id += 1

            # Calculate velocity if we have history
            if person['id'] in self.person_history and len(self.person_history[person['id']]) > 0:
                velocity = self._calculate_velocity(person, self.person_history[person['id']])
                person['velocity'] = velocity
                person['is_approaching'] = velocity > self.approach_velocity_threshold
            else:
                person['velocity'] = 0
                person['is_approaching'] = False

            # Update history
            if person['id'] not in self.person_history:
                self.person_history[person['id']] = deque(maxlen=10)

            self.person_history[person['id']].append(person)

            matched_persons.append(person)

        # Clean old history
        self._clean_old_history(current_time)

        return matched_persons

    def _calculate_velocity(self, current_person, history):
        """
        Calculate approach velocity (positive = approaching)

        Args:
            current_person: Current detection
            history: Deque of past detections

        Returns:
            float: Velocity in pixels/frame (positive = approaching)
        """
        if len(history) < 2:
            return 0

        # Get previous position
        prev = history[-1]
        curr_bbox = current_person['bbox']
        prev_bbox = prev['bbox']

        # Calculate change in bounding box size (larger = closer)
        curr_area = curr_bbox[2] * curr_bbox[3]
        prev_area = prev_bbox[2] * prev_bbox[3]

        # Velocity = change in size (approaching if increasing)
        velocity = (curr_area - prev_area) / max(prev_area, 1)

        # Convert to pixels/frame equivalent
        velocity_px = velocity * 100  # Scale factor

        return velocity_px

    def _clean_old_history(self, current_time):
        """Remove old tracking history"""
        timeout = 2.0  # seconds

        to_remove = []
        for person_id, history in self.person_history.items():
            if len(history) == 0:
                to_remove.append(person_id)
                continue

            last_time = history[-1]['timestamp']
            if current_time - last_time > timeout:
                to_remove.append(person_id)

        for person_id in to_remove:
            del self.person_history[person_id]

    def _assess_threat_level(self, person):
        """
        Assess threat level based on distance and approach
        MUCH LESS SENSITIVE - only alert when EXTREMELY close

        Args:
            person: Person detection dict

        Returns:
            str: 'critical', 'warning', 'safe'
        """
        bbox = person['bbox']
        height = bbox[3]
        distance_category = person['distance_category']
        is_approaching = person.get('is_approaching', False)
        velocity = person.get('velocity', 0)

        # CRITICAL: VERY close proximity (MUCH STRICTER)
        # Person must be REALLY close (height >250px) to trigger critical
        if distance_category == 'near' and height > 250:  # INCREASED from 200
            return 'critical'

        # WARNING: Approaching VERY rapidly (MUCH STRICTER)
        # Only alert if approaching extremely fast
        if is_approaching and velocity > self.approach_velocity_threshold * 2:  # 2x threshold (was 1.5x)
            if distance_category == 'near':  # Only if near (removed medium)
                return 'warning'

        # WARNING: Very close proximity (STRICTER - only near, not medium)
        if distance_category == 'near':
            return 'warning'

        # Everything else is SAFE (no alerts for medium/far distance)
        return 'safe'

    def should_alert(self, person):
        """
        Determine if we should trigger audio alert for this person
        Uses cooldown to prevent spam

        Args:
            person: Person detection dict

        Returns:
            bool: True if should alert
        """
        threat_level = person.get('threat_level', 'safe')

        # Only alert for critical or warning
        if threat_level not in ['critical', 'warning']:
            return False

        # Check cooldown
        person_id = person.get('id', -1)
        current_time = time.time()

        if person_id in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[person_id]
            if time_since_last < self.alert_cooldown:
                return False  # Still on cooldown

        # Update alert time
        self.last_alert_time[person_id] = current_time

        return True

    def get_alert_message(self, person):
        """
        Generate appropriate alert message

        Args:
            person: Person detection dict

        Returns:
            tuple: (message, priority)
        """
        zone = person['zone']
        threat_level = person['threat_level']
        orientation = person['orientation']
        is_approaching = person.get('is_approaching', False)

        # Extract direction from zone
        direction_map = {
            'left': 'left',
            'center': 'ahead',
            'right': 'right'
        }

        h_zone = zone.split('-')[0]
        direction = direction_map.get(h_zone, 'nearby')

        # Build message
        if threat_level == 'critical':
            if is_approaching:
                message = f"Warning! Person very close on your {direction}, approaching fast"
            else:
                message = f"Person very close on your {direction}"
            priority = 'critical'

        elif threat_level == 'warning':
            if is_approaching:
                message = f"Person approaching from your {direction}"
            else:
                message = f"Person nearby on your {direction}"
            priority = 'high'

        else:
            message = f"Person detected {direction}"
            priority = 'normal'

        return message, priority

    def visualize_detection(self, frame, detection_result):
        """
        Draw detection results on frame

        Args:
            frame: Original frame
            detection_result: Result from detect()

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for person in detection_result['persons']:
            x, y, w, h = person['bbox']
            threat_level = person['threat_level']

            # Color based on threat level
            if threat_level == 'critical':
                color = (0, 0, 255)  # Red
                thickness = 3
            elif threat_level == 'warning':
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (0, 255, 0)  # Green
                thickness = 1

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, thickness)

            # Build label
            label_parts = []
            label_parts.append(f"{person['confidence']:.2f}")
            label_parts.append(person['distance_category'])

            if person.get('is_approaching'):
                label_parts.append("APPROACHING")

            label = " | ".join(label_parts)

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(annotated, (x, y-label_h-10), (x+label_w, y), color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw orientation indicator
            if person['orientation'] != 'unknown':
                cv2.putText(annotated, person['orientation'], (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw approach arrow if approaching
            if person.get('is_approaching'):
                center_x = x + w // 2
                center_y = y + h // 2
                arrow_end_y = center_y + int(person.get('velocity', 0))
                cv2.arrowedLine(annotated, (center_x, center_y),
                              (center_x, arrow_end_y), (0, 0, 255), 2)

        # Summary info
        cv2.putText(annotated, f"Persons: {detection_result['total_persons']}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if detection_result['critical_alerts']:
            cv2.putText(annotated, f"CRITICAL: {len(detection_result['critical_alerts'])}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if detection_result['warnings']:
            cv2.putText(annotated, f"Warnings: {len(detection_result['warnings'])}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return annotated

    def get_average_inference_time(self):
        """Get average inference time in milliseconds"""
        if len(self.inference_times) == 0:
            return 0.0
        return np.mean(list(self.inference_times)) * 1000


# Standalone test function
def test_human_detector():
    """Test human detector with webcam"""
    print("=" * 60)
    print("Human Detector - Standalone Test")
    print("=" * 60)
    print("Controls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("=" * 60)
    print()

    # Initialize detector
    detector = HumanDetector(
        confidence_threshold=0.5,
        alert_distance_threshold=100,
        warning_distance_threshold=200,
        approach_velocity_threshold=15
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("📹 Camera opened")
    print("💡 Walk towards camera to test proximity detection")
    print()

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect humans
            result = detector.detect(frame)

            # Visualize
            annotated = detector.visualize_detection(frame, result)

            # Add performance info
            avg_time = detector.get_average_inference_time()
            cv2.putText(annotated, f"Inference: {avg_time:.1f}ms",
                       (10, annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show result
            cv2.imshow("Human Detector Test", annotated)

            # Check for alerts
            for person in result['critical_alerts']:
                if detector.should_alert(person):
                    message, priority = detector.get_alert_message(person)
                    print(f"🚨 {message}")

            for person in result['warnings']:
                if detector.should_alert(person):
                    message, priority = detector.get_alert_message(person)
                    print(f"⚠️ {message}")

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"human_detection_{frame_count}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"💾 Saved: {filename}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Test complete")
        print(f"Average inference time: {detector.get_average_inference_time():.1f}ms")


if __name__ == "__main__":
    test_human_detector()