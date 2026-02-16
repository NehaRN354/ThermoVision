"""
ThermoVision - Configuration Settings
Central configuration for all system parameters
"""

import os


class Config:
    """System-wide configuration"""

    # ================== LOW-LIGHT ENHANCEMENT (NOVELTY FEATURE) ==================
    ENABLE_LOW_LIGHT_ENHANCEMENT = True
    LOW_LIGHT_THRESHOLD = 80  # Mean brightness below this = low light
    VERY_DARK_THRESHOLD = 40  # Very dark conditions
    ENHANCEMENT_LEVEL = 2.0   # 1.0-3.0 (strength of enhancement)

    # ================== CAMERA SETTINGS ==================
    CAMERA_ID = 0  # Default webcam
    CAMERA_RESOLUTION = (640, 480)  # Width x Height
    TARGET_FPS = 30

    # ================== DETECTION SETTINGS ==================
    # Fire Detection
    FIRE_CONFIDENCE_THRESHOLD = 0.50  # 50% confidence (LOWERED for demo - was 0.65)
    FIRE_COLOR_LOWER = (0, 100, 100)  # HSV lower bound (MORE SENSITIVE - was 120)
    FIRE_COLOR_UPPER = (30, 255, 255)  # HSV upper bound (WIDER range - was 25)

    # Human Detection
    HUMAN_CONFIDENCE_THRESHOLD = 0.5  # 50% confidence minimum
    YOLO_MODEL = "yolov8n.pt"  # Nano model for speed (can upgrade to yolov8s.pt for accuracy)

    # Proximity thresholds (in pixels - based on person height in frame)
    HUMAN_ALERT_DISTANCE = 200    # <200px height = CRITICAL (INCREASED - was 150)
    HUMAN_WARNING_DISTANCE = 300  # <300px height = WARNING (INCREASED - was 250)
    HUMAN_SAFE_DISTANCE = 300     # >300px height = SAFE

    # Approach detection
    APPROACH_VELOCITY_THRESHOLD = 35  # pixels/frame - fast approach (INCREASED - was 25)

    # Hot Object Detection
    HOT_OBJECT_CLASSES = ['car', 'motorcycle', 'truck', 'bus', 'bicycle', 'oven', 'microwave']
    HOT_OBJECT_CONFIDENCE = 0.45  # LOWERED from 0.6 for demo sensitivity

    # Vehicle detection thresholds
    VEHICLE_ALERT_DISTANCE = 180    # LOWERED from 200 (more sensitive)
    VEHICLE_WARNING_DISTANCE = 280  # LOWERED from 300 (more sensitive)
    MOVING_VEHICLE_THRESHOLD = 20   # LOWERED from 25 (more sensitive)

    # ================== RISK ASSESSMENT ==================
    # Distance zones (in pixels - approximate)
    ZONE_CRITICAL = 150   # < 150px = Critical danger
    ZONE_WARNING = 300    # 150-300px = Warning
    ZONE_SAFE = 300       # > 300px = Safe

    # Risk weights
    WEIGHT_FIRE = 1.0         # Highest priority
    WEIGHT_HOT_OBJECT = 0.7   # Medium-high priority
    WEIGHT_HUMAN = 0.5        # Medium priority
    WEIGHT_POSE_UNSTABLE = 0.3  # Additional risk factor

    # Risk thresholds
    RISK_THRESHOLD_DANGER = 0.7   # 70% = Red alert
    RISK_THRESHOLD_CAUTION = 0.4  # 40% = Yellow alert

    # ================== AUDIO SETTINGS ==================
    ENABLE_AUDIO = True
    SPEECH_RATE = 150  # Words per minute
    AUDIO_COOLDOWN = 3.0  # 3 SECONDS between alerts for same object/person

    # Alert messages
    ALERT_FIRE = "Fire detected ahead"
    ALERT_HOT_OBJECT = "Hot object detected"
    ALERT_HUMAN_APPROACHING = "Person approaching"
    ALERT_DANGER_ZONE = "Danger - stop immediately"

    # ================== MEDIAPIPE SETTINGS ==================
    POSE_CONFIDENCE = 0.5
    POSE_TRACKING_CONFIDENCE = 0.5
    ENABLE_POSE_DETECTION = True

    # Instability thresholds
    STABILITY_THRESHOLD = 0.15  # Hip/shoulder movement variance

    # ================== DISPLAY SETTINGS ==================
    ENABLE_VISUALIZATION = True  # Show debug window
    SHOW_FPS = True
    SHOW_DETECTIONS = True
    SHOW_ZONES = True

    # Colors for visualization (BGR format)
    COLOR_SAFE = (0, 255, 0)      # Green
    COLOR_CAUTION = (0, 255, 255) # Yellow
    COLOR_DANGER = (0, 0, 255)    # Red
    COLOR_FIRE = (0, 100, 255)    # Orange
    COLOR_HUMAN = (255, 0, 0)     # Blue

    # ================== PERFORMANCE SETTINGS ==================
    MAX_DETECTION_TIME = 0.2  # Max 200ms per frame
    SKIP_FRAMES = 2  # Process every Nth frame (1 = all frames, 2 = every other)

    # ================== FILE PATHS ==================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    # Model paths
    FIRE_MODEL_PATH = os.path.join(MODELS_DIR, 'fire_detector.pth')
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, YOLO_MODEL)

    # ================== LOGGING ==================
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("ThermoVision Configuration")
        print("=" * 60)
        print(f"Camera: {cls.CAMERA_RESOLUTION[0]}x{cls.CAMERA_RESOLUTION[1]} @ {cls.TARGET_FPS} FPS")
        print(f"YOLO Model: {cls.YOLO_MODEL}")
        print(f"Fire Threshold: {cls.FIRE_CONFIDENCE_THRESHOLD}")
        print(f"Human Threshold: {cls.HUMAN_CONFIDENCE_THRESHOLD}")
        print(f"Audio: {'Enabled' if cls.ENABLE_AUDIO else 'Disabled'}")
        print(f"Visualization: {'Enabled' if cls.ENABLE_VISUALIZATION else 'Disabled'}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()