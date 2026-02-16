"""
ThermoVision - Hot Object Detection Module
Detects vehicles and hot objects for road safety
Focus: Cars, trucks, bikes, motorcycles + kitchen appliances
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time


class HotObjectDetector:
    """
    Detect hot/dangerous objects with emphasis on vehicles for road safety

    Vehicle Categories:
    - Cars, trucks, buses (hot engines, collision risk)
    - Motorcycles, bicycles (fast-moving, collision risk)

    Other Hot Objects:
    - Kitchen: oven, stove, microwave, toaster
    - Electronics: laptop, TV (can overheat)
    """

    def __init__(self,
                 confidence_threshold=0.5,
                 vehicle_alert_distance=150,  # pixels - vehicle proximity alert
                 moving_vehicle_threshold=20,  # pixels/frame - motion detection
                 high_risk_distance=100):      # pixels - critical proximity
        """
        Initialize hot object detector

        Args:
            confidence_threshold: Minimum confidence for detection
            vehicle_alert_distance: Distance threshold for vehicle alerts
            moving_vehicle_threshold: Speed threshold for moving vehicles
            high_risk_distance: Critical proximity threshold
        """
        self.confidence_threshold = confidence_threshold
        self.vehicle_alert_distance = vehicle_alert_distance
        self.moving_vehicle_threshold = moving_vehicle_threshold
        self.high_risk_distance = high_risk_distance

        # Initialize YOLO model (same as human detector for efficiency)
        print("🚗 Loading YOLO model for object detection...")
        try:
            self.model = YOLO('yolov8n.pt')  # Nano model for real-time
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠️ YOLO model download in progress: {e}")
            self.model = YOLO('yolov8n.pt')

        # COCO class IDs for vehicles and hot objects
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }

        # ONLY truly hot appliances (REMOVED: dining table, keyboard, phone, sink)
        self.appliance_classes = {
            69: 'oven',
            72: 'microwave',
            # Removed unnecessary items for cleaner detection
        }

        # Heat risk levels per object type
        self.heat_risk = {
            'car': 0.8,
            'truck': 0.9,
            'bus': 0.9,
            'motorcycle': 0.7,
            'bicycle': 0.3,
            'oven': 1.0,
            'microwave': 0.6
        }

        # Object tracking for motion detection
        self.object_history = {}  # {object_id: deque of positions}
        self.next_object_id = 0

        # Alert cooldown
        self.last_alert_time = {}
        self.alert_cooldown = 4.0  # seconds

        # Performance tracking
        self.inference_times = deque(maxlen=30)

        print("✅ Hot object detector initialized")
        print(f"   Tracking: {len(self.vehicle_classes)} vehicle types")
        print(f"   Tracking: {len(self.appliance_classes)} appliance types")
        print(f"   Vehicle alert distance: <{vehicle_alert_distance}px")

    def detect(self, frame):
        """
        Detect hot objects and vehicles

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict: {
                'vehicles': list of vehicle detections,
                'appliances': list of appliance detections,
                'critical_alerts': list of high-risk objects,
                'warnings': list of moderate-risk objects,
                'total_objects': int,
                'inference_time': float
            }
        """
        start_time = time.time()

        # Run YOLO detection
        results = self.model(frame, verbose=False)

        # Extract detections
        vehicles = []
        appliances = []

        frame_height, frame_width = frame.shape[:2]

        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Check confidence
                if confidence < self.confidence_threshold:
                    continue

                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

                # Categorize object
                object_type = None
                category = None

                if class_id in self.vehicle_classes:
                    object_type = self.vehicle_classes[class_id]
                    category = 'vehicle'
                elif class_id in self.appliance_classes:
                    object_type = self.appliance_classes[class_id]
                    category = 'appliance'
                else:
                    continue  # Not a hot object we care about

                # Calculate properties
                distance_score = self._estimate_distance(h, frame_height)
                zone = self._get_position_zone(x, y, w, h, frame_width, frame_height)
                heat_level = self.heat_risk.get(object_type, 0.5)

                obj = {
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'type': object_type,
                    'category': category,
                    'class_id': class_id,
                    'distance_score': distance_score,
                    'distance_category': self._categorize_distance(distance_score),
                    'zone': zone,
                    'center': (x + w//2, y + h//2),
                    'heat_level': heat_level,
                    'timestamp': time.time()
                }

                if category == 'vehicle':
                    vehicles.append(obj)
                else:
                    appliances.append(obj)

        # Track objects and detect motion (especially vehicles)
        vehicles = self._track_and_calculate_motion(vehicles)
        appliances = self._track_and_calculate_motion(appliances)

        # Assess threat levels
        critical_alerts = []
        warnings = []

        all_objects = vehicles + appliances

        for obj in all_objects:
            threat_level = self._assess_threat_level(obj)
            obj['threat_level'] = threat_level

            if threat_level == 'critical':
                critical_alerts.append(obj)
            elif threat_level == 'warning':
                warnings.append(obj)

        # Performance tracking
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        result = {
            'vehicles': vehicles,
            'appliances': appliances,
            'critical_alerts': critical_alerts,
            'warnings': warnings,
            'total_objects': len(all_objects),
            'inference_time': inference_time
        }

        return result

    def _estimate_distance(self, object_height, frame_height):
        """
        Estimate distance based on object height in frame

        Args:
            object_height: Height of object bounding box (pixels)
            frame_height: Total frame height

        Returns:
            float: Distance score (0-1, where 1 = very close)
        """
        height_ratio = object_height / frame_height
        distance_score = min(height_ratio * 2, 1.0)
        return distance_score

    def _categorize_distance(self, distance_score):
        """Categorize distance into near/medium/far"""
        if distance_score > 0.5:
            return 'near'
        elif distance_score > 0.25:
            return 'medium'
        else:
            return 'far'

    def _get_position_zone(self, x, y, w, h, frame_width, frame_height):
        """Determine position zone (left/center/right)"""
        center_x = x + w // 2

        if center_x < frame_width / 3:
            h_zone = 'left'
        elif center_x < 2 * frame_width / 3:
            h_zone = 'center'
        else:
            h_zone = 'right'

        return h_zone

    def _track_and_calculate_motion(self, objects):
        """
        Track objects and calculate motion (important for vehicles)

        Args:
            objects: List of detected objects

        Returns:
            List of objects with motion information
        """
        current_time = time.time()

        # Match current detections with tracked objects
        matched_objects = []

        for obj in objects:
            center = obj['center']
            obj_type = obj['type']

            # Find closest tracked object of same type
            best_match_id = None
            best_distance = float('inf')

            for obj_id, history in self.object_history.items():
                if len(history) == 0:
                    continue

                # Check if same type
                if history[-1]['type'] != obj_type:
                    continue

                last_center = history[-1]['center']
                dist = np.sqrt(
                    (center[0] - last_center[0])**2 +
                    (center[1] - last_center[1])**2
                )

                if dist < 150 and dist < best_distance:
                    best_distance = dist
                    best_match_id = obj_id

            # Assign ID
            if best_match_id is not None:
                obj['id'] = best_match_id
            else:
                obj['id'] = self.next_object_id
                self.next_object_id += 1

            # Calculate motion if we have history
            if obj['id'] in self.object_history and len(self.object_history[obj['id']]) > 0:
                motion_info = self._calculate_motion(obj, self.object_history[obj['id']])
                obj.update(motion_info)
            else:
                obj['is_moving'] = False
                obj['motion_speed'] = 0
                obj['motion_direction'] = 'stationary'

            # Update history
            if obj['id'] not in self.object_history:
                self.object_history[obj['id']] = deque(maxlen=10)

            self.object_history[obj['id']].append(obj)

            matched_objects.append(obj)

        # Clean old history
        self._clean_old_history(current_time)

        return matched_objects

    def _calculate_motion(self, current_obj, history):
        """
        Calculate motion for vehicles (important for road safety)

        Args:
            current_obj: Current detection
            history: Deque of past detections

        Returns:
            dict: Motion information
        """
        if len(history) < 2:
            return {
                'is_moving': False,
                'motion_speed': 0,
                'motion_direction': 'stationary'
            }

        prev = history[-1]
        curr_center = current_obj['center']
        prev_center = prev['center']

        # Calculate displacement
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]

        # Speed (pixels/frame)
        speed = np.sqrt(dx**2 + dy**2)

        # Direction
        if speed > 5:  # Moving threshold
            if abs(dx) > abs(dy):
                direction = 'horizontal' if dx > 0 else 'horizontal'
            else:
                direction = 'approaching' if dy > 0 else 'receding'
            is_moving = True
        else:
            direction = 'stationary'
            is_moving = False

        return {
            'is_moving': is_moving,
            'motion_speed': speed,
            'motion_direction': direction
        }

    def _clean_old_history(self, current_time):
        """Remove old tracking history"""
        timeout = 3.0  # seconds

        to_remove = []
        for obj_id, history in self.object_history.items():
            if len(history) == 0:
                to_remove.append(obj_id)
                continue

            last_time = history[-1]['timestamp']
            if current_time - last_time > timeout:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.object_history[obj_id]

    def _assess_threat_level(self, obj):
        """
        Assess threat level based on object type, distance, heat, and motion
        MORE SENSITIVE for demo

        Args:
            obj: Object detection dict

        Returns:
            str: 'critical', 'warning', 'safe'
        """
        obj_type = obj['type']
        category = obj['category']
        distance_category = obj['distance_category']
        heat_level = obj['heat_level']
        is_moving = obj.get('is_moving', False)
        motion_speed = obj.get('motion_speed', 0)
        bbox = obj['bbox']
        height = bbox[3]

        # CRITICAL: Very close vehicle (MORE SENSITIVE)
        if category == 'vehicle' and distance_category == 'near' and height > 200:  # LOWERED from 250
            return 'critical'

        # CRITICAL: Fast moving vehicle close by (MORE SENSITIVE)
        if category == 'vehicle' and is_moving and motion_speed > self.moving_vehicle_threshold:
            if distance_category in ['near', 'medium']:
                return 'critical'

        # CRITICAL: Very hot appliance very nearby (MORE SENSITIVE)
        if category == 'appliance' and heat_level >= 0.8:  # LOWERED from 0.9
            if distance_category == 'near' and height > 150:  # LOWERED from 200
                return 'critical'

        # WARNING: Vehicle in close range (MORE SENSITIVE)
        if category == 'vehicle' and distance_category in ['near', 'medium']:  # Added medium
            return 'warning'

        # WARNING: Hot appliance detected close (MORE SENSITIVE)
        if category == 'appliance' and heat_level >= 0.6:  # LOWERED from 0.8
            if distance_category in ['near', 'medium']:
                return 'warning'

        return 'safe'

    def should_alert(self, obj):
        """
        Determine if we should trigger audio alert
        Uses cooldown to prevent spam

        Args:
            obj: Object detection dict

        Returns:
            bool: True if should alert
        """
        threat_level = obj.get('threat_level', 'safe')

        # Only alert for critical or warning
        if threat_level not in ['critical', 'warning']:
            return False

        # Check cooldown
        obj_id = obj.get('id', -1)
        current_time = time.time()

        if obj_id in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[obj_id]
            if time_since_last < self.alert_cooldown:
                return False

        # Update alert time
        self.last_alert_time[obj_id] = current_time

        return True

    def get_alert_message(self, obj):
        """
        Generate appropriate alert message

        Args:
            obj: Object detection dict

        Returns:
            tuple: (message, priority)
        """
        obj_type = obj['type']
        category = obj['category']
        zone = obj['zone']
        threat_level = obj['threat_level']
        is_moving = obj.get('is_moving', False)

        # Direction
        direction_map = {
            'left': 'left',
            'center': 'ahead',
            'right': 'right'
        }
        direction = direction_map.get(zone, 'nearby')

        # Build message
        if category == 'vehicle':
            if threat_level == 'critical':
                if is_moving:
                    message = f"Warning! Moving {obj_type} approaching from {direction}"
                else:
                    message = f"Vehicle very close on your {direction}"
                priority = 'critical'
            else:
                message = f"{obj_type.capitalize()} detected {direction}"
                priority = 'high'

        else:  # appliance
            if threat_level == 'critical':
                message = f"Warning! Hot {obj_type} very close on your {direction}"
                priority = 'critical'
            else:
                message = f"Hot {obj_type} detected {direction}"
                priority = 'high'

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

        all_objects = detection_result['vehicles'] + detection_result['appliances']

        for obj in all_objects:
            x, y, w, h = obj['bbox']
            threat_level = obj['threat_level']
            obj_type = obj['type']
            category = obj['category']

            # Color based on threat level
            if threat_level == 'critical':
                color = (0, 0, 255)  # Red
                thickness = 3
            elif threat_level == 'warning':
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (255, 255, 0)  # Yellow (detected but safe)
                thickness = 1

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, thickness)

            # Build label
            label = f"{obj_type} {obj['confidence']:.2f}"

            if obj.get('is_moving'):
                label += " [MOVING]"

            # Heat indicator for appliances
            if category == 'appliance':
                heat = int(obj['heat_level'] * 100)
                label += f" {heat}°"

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(annotated, (x, y-label_h-10), (x+label_w, y), color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw category icon
            icon_text = "🚗" if category == 'vehicle' else "🔥"
            cv2.putText(annotated, icon_text, (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Summary info
        cv2.putText(annotated, f"Vehicles: {len(detection_result['vehicles'])}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Appliances: {len(detection_result['appliances'])}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if detection_result['critical_alerts']:
            cv2.putText(annotated, f"CRITICAL: {len(detection_result['critical_alerts'])}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated

    def get_average_inference_time(self):
        """Get average inference time in milliseconds"""
        if len(self.inference_times) == 0:
            return 0.0
        return np.mean(list(self.inference_times)) * 1000


# Standalone test function
def test_hot_object_detector():
    """Test hot object detector with webcam"""
    print("=" * 60)
    print("Hot Object Detector - Standalone Test")
    print("=" * 60)
    print("Testing vehicle and appliance detection")
    print()
    print("Controls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("=" * 60)
    print()

    # Initialize detector
    detector = HotObjectDetector(
        confidence_threshold=0.5,
        vehicle_alert_distance=150,
        moving_vehicle_threshold=20
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Camera opened")
    print("💡 Show toy cars, laptop, or go outside for real vehicles")
    print()

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            result = detector.detect(frame)

            # Visualize
            annotated = detector.visualize_detection(frame, result)

            # Add performance info
            avg_time = detector.get_average_inference_time()
            cv2.putText(annotated, f"Inference: {avg_time:.1f}ms",
                       (10, annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show result
            cv2.imshow("Hot Object Detector Test", annotated)

            # Print alerts
            for obj in result['critical_alerts']:
                if detector.should_alert(obj):
                    message, priority = detector.get_alert_message(obj)
                    print(f"🚨 {message}")

            for obj in result['warnings']:
                if detector.should_alert(obj):
                    message, priority = detector.get_alert_message(obj)
                    print(f"⚠️ {message}")

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"hot_object_{frame_count}.jpg"
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
    test_hot_object_detector()