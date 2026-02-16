"""
ThermoVision - Main Application
Integrates all modules into complete hazard detection system
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from camera.camera_input import CameraInput
from audio.audio_feedback import AudioFeedback
from utils.config import Config


class ThermoVision:
    """
    Main ThermoVision application
    Coordinates camera, detection, and audio feedback
    """

    def __init__(self):
        """Initialize ThermoVision system"""
        print("=" * 60)
        print("ThermoVision - Initializing...")
        print("=" * 60)
        print()

        # Print configuration
        Config.print_config()
        print()

        # Initialize modules
        self.camera = None
        self.audio = None
        self.low_light_enhancer = None  # NEW: Low-light enhancement

        # Detection modules
        self.fire_detector = None
        self.human_detector = None
        self.hot_object_detector = None

        self._initialize_detectors()

        # System state
        self.is_running = False
        self.frame_count = 0
        self.skip_counter = 0

        # Current detections
        self.current_hazards = {
            'fire': [],
            'humans': [],
            'hot_objects': []
        }

        # Current alert messages (for on-screen display)
        self.current_alert_messages = []
        self.alert_message_time = {}  # Track when each message was shown

        # Risk state
        self.current_risk_level = "safe"  # "safe", "caution", "danger"

        self._initialize_system()

    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # ==============================
            # CAMERA SELECTION (NEW)
            # ==============================

            print("Select Camera Source:")
            print("1 - Laptop Webcam")
            print("2 - Phone Camera (DroidCam)")
            print()

            choice = input("Enter choice (1 or 2): ").strip()

            if choice == "2":
                print("\nChoose DroidCam connection type:")
                print("1 - USB / Virtual Camera (camera index like 1 or 2)")
                print("2 - WiFi / IP Camera (URL)")

                sub_choice = input("Enter choice (1 or 2): ").strip()

                if sub_choice == "1":
                    cam_index = input(
                        "Enter camera index (usually 1 or 2): ").strip()

                    camera_source = int(cam_index)

                else:
                    camera_source = input(
                        "Enter DroidCam URL (example: http://192.168.1.6:4747/video): ").strip()

            else:
                camera_source = Config.CAMERA_ID

            # ==============================
            # Initialize camera
            # ==============================

            print("\nInitializing camera...")

            self.camera = CameraInput(
                camera_id=camera_source,  # <-- NOW dynamic
                resolution=Config.CAMERA_RESOLUTION,
                fps_target=Config.TARGET_FPS
            )

            # Initialize audio
            if Config.ENABLE_AUDIO:
                print("Initializing audio feedback...")
                self.audio = AudioFeedback(
                    rate=Config.SPEECH_RATE,
                    cooldown=Config.AUDIO_COOLDOWN
                )

            print()
            print("✅ System initialized successfully")
            print()

        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise

    def _initialize_detectors(self):
        """Initialize AI detection modules"""
        try:
            # Initialize low-light enhancer FIRST
            try:
                from camera.low_light_enhancer import LowLightEnhancer
                print("Initializing low-light enhancer...")
                self.low_light_enhancer = LowLightEnhancer(
                    enable_auto=True,
                    enhancement_level=2.0
                )
            except Exception as e:
                print(f"⚠️ Low-light enhancer initialization failed: {e}")

            # TODO: Initialize detection modules in Phase 1
            # Import here to avoid circular imports
            try:
                from detection.fire_detector import FireDetector
                print("Initializing fire detector...")
                self.fire_detector = FireDetector(
                    confidence_threshold=Config.FIRE_CONFIDENCE_THRESHOLD
                )
            except Exception as e:
                print(f"⚠️ Fire detector initialization failed: {e}")

            try:
                from detection.human_detector import HumanDetector
                print("Initializing human detector...")
                self.human_detector = HumanDetector(
                    confidence_threshold=Config.HUMAN_CONFIDENCE_THRESHOLD,
                    alert_distance_threshold=Config.HUMAN_ALERT_DISTANCE,
                    warning_distance_threshold=Config.HUMAN_WARNING_DISTANCE,
                    approach_velocity_threshold=Config.APPROACH_VELOCITY_THRESHOLD
                )
            except Exception as e:
                print(f"⚠️ Human detector initialization failed: {e}")

            try:
                from detection.hot_object_detector import HotObjectDetector
                print("Initializing hot object detector...")
                self.hot_object_detector = HotObjectDetector(
                    confidence_threshold=Config.HOT_OBJECT_CONFIDENCE,
                    vehicle_alert_distance=Config.VEHICLE_ALERT_DISTANCE,
                    moving_vehicle_threshold=Config.MOVING_VEHICLE_THRESHOLD
                )
            except Exception as e:
                print(f"⚠️ Hot object detector initialization failed: {e}")

            print()
            print("✅ System initialized successfully")
            print()

        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise

    def run(self):
        """Main application loop"""
        self.is_running = True

        print("=" * 60)
        print("🚀 ThermoVision RUNNING")
        print("=" * 60)
        print("Controls:")
        print("  Q - Quit")
        print("  S - Toggle stabilization")
        print("  A - Toggle audio")
        print("  D - Toggle debug view")
        print("=" * 60)
        print()

        if self.audio:
            self.audio.speak("ThermoVision started", priority="low")

        try:
            while self.is_running:
                # Process frame
                success = self._process_frame()

                if not success:
                    print("⚠️ Frame processing failed")
                    break

                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Shutting down...")
                    break
                elif key == ord('a'):
                    Config.ENABLE_AUDIO = not Config.ENABLE_AUDIO
                    status = "ON" if Config.ENABLE_AUDIO else "OFF"
                    print(f"Audio: {status}")
                elif key == ord('d'):
                    Config.ENABLE_VISUALIZATION = not Config.ENABLE_VISUALIZATION
                    print(f"Debug view: {'ON' if Config.ENABLE_VISUALIZATION else 'OFF'}")

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")

        finally:
            self._shutdown()

    def _process_frame(self):
        """
        Process single frame through detection pipeline

        Returns:
            bool: Success status
        """
        # Read frame from camera
        ret, processed_frame, original_frame = self.camera.get_frame_for_detection()

        if not ret:
            return False

        # === LOW-LIGHT ENHANCEMENT (KEY NOVELTY) ===
        enhancement_metadata = {'enhanced': False}
        if self.low_light_enhancer:
            processed_frame, enhancement_metadata = self.low_light_enhancer.enhance(processed_frame)

            # Update original frame for display
            if enhancement_metadata['enhanced']:
                original_frame = processed_frame.copy()

        # Frame skipping for performance (DISABLED for equal detection)
        # Process EVERY frame to ensure all detectors run equally
        self.skip_counter += 1
        if self.skip_counter < 1:  # Changed from Config.SKIP_FRAMES to 1 (no skipping)
            # Just display, don't process
            if Config.ENABLE_VISUALIZATION:
                self._display_frame(original_frame, skip=True)
            return True

        self.skip_counter = 0

        # === DETECTION PIPELINE ===
        # Clear old alert messages (keep for 3 seconds)
        current_time = time.time()
        self.current_alert_messages = [
            msg for msg in self.current_alert_messages
            if current_time - self.alert_message_time.get(msg, 0) < 3.0
        ]

        # Reset risk level at start of each frame
        self.current_risk_level = "safe"

        # Fire detection
        if self.fire_detector:
            fire_result = self.fire_detector.detect(processed_frame)

            # Update hazards
            if fire_result['fire_detected']:
                self.current_hazards['fire'] = fire_result['bounding_boxes']
                self.current_risk_level = "danger"  # Fire = immediate danger

                # Audio alert (only if cooldown allows)
                if fire_result.get('should_alert', False):
                    alert_msg = "FIRE DETECTED AHEAD"

                    if Config.ENABLE_AUDIO and self.audio:
                        self.audio.alert_fire(direction="ahead", distance="close")

                    # Add to on-screen messages
                    if alert_msg not in self.current_alert_messages:
                        self.current_alert_messages.append(alert_msg)
                        self.alert_message_time[alert_msg] = current_time
            else:
                self.current_hazards['fire'] = []

        # Human detection
        if self.human_detector:
            human_result = self.human_detector.detect(processed_frame)

            # Update hazards
            self.current_hazards['humans'] = [
                p['bbox'] for p in human_result['persons']
            ]

            # Process alerts based on proximity
            for person in human_result['critical_alerts']:
                if self.human_detector.should_alert(person):
                    message, priority = self.human_detector.get_alert_message(person)

                    # Audio alert
                    if Config.ENABLE_AUDIO and self.audio:
                        self.audio.alert_custom(message, priority=priority)

                    # Add to on-screen messages (no emojis)
                    alert_msg = f"ALERT: {message.upper()}"
                    if alert_msg not in self.current_alert_messages:
                        self.current_alert_messages.append(alert_msg)
                        self.alert_message_time[alert_msg] = current_time

                    # Update risk level (only if not already danger from fire)
                    if self.current_risk_level != "danger":
                        self.current_risk_level = "danger"

            for person in human_result['warnings']:
                if self.human_detector.should_alert(person):
                    message, priority = self.human_detector.get_alert_message(person)

                    # Audio alert (lower priority)
                    if Config.ENABLE_AUDIO and self.audio:
                        self.audio.alert_custom(message, priority=priority)

                    # Add to on-screen messages (no emojis)
                    alert_msg = f"WARNING: {message}"
                    if alert_msg not in self.current_alert_messages:
                        self.current_alert_messages.append(alert_msg)
                        self.alert_message_time[alert_msg] = current_time

                    # Update risk level (only if currently safe)
                    if self.current_risk_level == "safe":
                        self.current_risk_level = "caution"

        # Hot object detection (vehicles + appliances)
        if self.hot_object_detector:
            hot_result = self.hot_object_detector.detect(processed_frame)

            # Update hazards
            self.current_hazards['hot_objects'] = [
                obj['bbox'] for obj in (hot_result['vehicles'] + hot_result['appliances'])
            ]

            # Process alerts
            for obj in hot_result['critical_alerts']:
                if self.hot_object_detector.should_alert(obj):
                    message, priority = self.hot_object_detector.get_alert_message(obj)

                    # Audio alert
                    if Config.ENABLE_AUDIO and self.audio:
                        self.audio.alert_custom(message, priority=priority)

                    # Add to on-screen messages (no emojis)
                    alert_msg = f"VEHICLE ALERT: {message.upper()}"
                    if alert_msg not in self.current_alert_messages:
                        self.current_alert_messages.append(alert_msg)
                        self.alert_message_time[alert_msg] = current_time

                    # Update risk level (only if not already danger)
                    if self.current_risk_level != "danger":
                        self.current_risk_level = "danger"

            for obj in hot_result['warnings']:
                if self.hot_object_detector.should_alert(obj):
                    message, priority = self.hot_object_detector.get_alert_message(obj)

                    # Audio alert
                    if Config.ENABLE_AUDIO and self.audio:
                        self.audio.alert_custom(message, priority=priority)

                    # Add to on-screen messages (no emojis)
                    alert_msg = f"VEHICLE: {message}"
                    if alert_msg not in self.current_alert_messages:
                        self.current_alert_messages.append(alert_msg)
                        self.alert_message_time[alert_msg] = current_time

                    # Update risk level (only if currently safe)
                    if self.current_risk_level == "safe":
                        self.current_risk_level = "caution"

        # NO MORE VISUAL OVERLAYS - Just display clean frame
        if Config.ENABLE_VISUALIZATION:
            display_frame = self._draw_debug_info(original_frame, enhancement_metadata)
            self._display_frame(display_frame, skip=False)

        # For now, just display frame
        if Config.ENABLE_VISUALIZATION:
            display_frame = self._draw_debug_info(original_frame)
            self._display_frame(display_frame, skip=False)

        return True

    def _draw_debug_info(self, frame, enhancement_metadata=None):
        """
        Draw PROFESSIONAL interface overlay
        Clean, no emojis, proper status logic

        Args:
            frame: Input frame
            enhancement_metadata: Low-light enhancement info

        Returns:
            Frame with professional overlay
        """
        display = frame.copy()
        height, width = display.shape[:2]

        # === HEADER BAR (Professional) ===
        header_height = 100
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (width, header_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)

        # Title
        cv2.putText(display, "ThermoVision", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display, "Hazard Detection System", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # FPS (top right)
        fps = self.camera.get_fps()
        cv2.putText(display, f"FPS: {fps}", (width - 120, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # === STATUS INDICATOR (Correct Logic) ===
        risk_colors = {
            "safe": (0, 255, 0),      # Green
            "caution": (0, 255, 255),  # Yellow
            "danger": (0, 0, 255)      # Red
        }

        risk_color = risk_colors.get(self.current_risk_level, (255, 255, 255))
        status_text = self.current_risk_level.upper()

        # Status box
        status_width = 200
        cv2.rectangle(display, (width - status_width - 20, 50),
                     (width - 20, 85), risk_color, 2)
        cv2.putText(display, f"STATUS: {status_text}", (width - status_width - 10, 73),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)

        # === LOW-LIGHT INDICATOR (No emojis) ===
        if enhancement_metadata and enhancement_metadata.get('enhanced', False):
            mode = enhancement_metadata['mode']
            brightness = enhancement_metadata['brightness']

            if mode == "very_dark":
                color = (0, 0, 255)  # Red
                text = "NIGHT MODE"
            else:
                color = (0, 165, 255)  # Orange
                text = "LOW LIGHT"

            # Low-light badge
            cv2.rectangle(display, (width - 180, 10), (width - 20, 40), color, 2)
            cv2.putText(display, text, (width - 170, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # === DETECTION COUNTERS (Bottom left) ===
        info_y = height - 100
        counter_x = 20

        fire_count = len(self.current_hazards['fire'])
        people_count = len(self.current_hazards['humans'])
        vehicle_count = len(self.current_hazards['hot_objects'])

        # Background for counters
        overlay = display.copy()
        cv2.rectangle(overlay, (10, height - 120), (250, height - 10), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)

        # Detection counts
        cv2.putText(display, "DETECTIONS:", (counter_x, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 25

        if fire_count > 0:
            cv2.putText(display, f"Fire: {fire_count}", (counter_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            info_y += 25

        if people_count > 0:
            cv2.putText(display, f"People: {people_count}", (counter_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            info_y += 25

        if vehicle_count > 0:
            cv2.putText(display, f"Vehicles: {vehicle_count}", (counter_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

        # === ALERT MESSAGES (Center - Professional, AUTO-SIZED) ===
        if self.current_alert_messages:
            # Calculate required height based on number of messages
            num_messages = min(len(self.current_alert_messages), 4)  # Max 4 messages
            alert_box_height = num_messages * 50 + 80  # 50px per message + padding
            alert_y_start = (height - alert_box_height) // 2

            # Calculate width based on longest message
            max_msg_length = max(len(msg) for msg in self.current_alert_messages)
            alert_box_width = min(max_msg_length * 12, width - 100)  # 12px per char, max screen width - margin
            alert_x_start = (width - alert_box_width) // 2

            # Alert box background (dark, semi-transparent)
            overlay = display.copy()
            cv2.rectangle(overlay, (alert_x_start, alert_y_start),
                         (alert_x_start + alert_box_width, alert_y_start + alert_box_height),
                         (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.9, display, 0.1, 0, display)

            # Alert box border
            cv2.rectangle(display, (alert_x_start, alert_y_start),
                         (alert_x_start + alert_box_width, alert_y_start + alert_box_height),
                         (0, 0, 255), 3)

            # "ALERTS" header
            cv2.putText(display, "ALERTS", (alert_x_start + 20, alert_y_start + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Alert messages (no emojis, clean text, AUTO-WRAP if needed)
            alert_y = alert_y_start + 70
            for msg in self.current_alert_messages[-4:]:  # Show last 4 alerts max
                # Truncate message if too long
                max_chars = (alert_box_width - 40) // 8  # Approximate chars that fit
                if len(msg) > max_chars:
                    msg = msg[:max_chars-3] + "..."

                # Determine color based on message type (no emojis to check)
                if "FIRE" in msg:
                    msg_color = (0, 100, 255)  # Orange
                elif "ALERT" in msg or ("WARNING" in msg and "VEHICLE" not in msg):
                    msg_color = (0, 0, 255)    # Red
                elif "VEHICLE" in msg:
                    msg_color = (100, 255, 255)  # Cyan
                else:
                    msg_color = (255, 255, 255)  # White

                # Draw text (clean, professional, smaller font)
                cv2.putText(display, msg, (alert_x_start + 20, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, msg_color, 1)
                alert_y += 50

        return display

    def _draw_detection_zones(self, frame):
        """Draw 3x3 detection zone grid"""
        height, width = frame.shape[:2]

        # Vertical lines
        cv2.line(frame, (width//3, 0), (width//3, height), (100, 100, 100), 1)
        cv2.line(frame, (2*width//3, 0), (2*width//3, height), (100, 100, 100), 1)

        # Horizontal lines
        cv2.line(frame, (0, height//3), (width, height//3), (100, 100, 100), 1)
        cv2.line(frame, (0, 2*height//3), (width, 2*height//3), (100, 100, 100), 1)

        # Zone labels
        zones = ['L-Far', 'C-Far', 'R-Far',
                'L-Mid', 'Center', 'R-Mid',
                'L-Near', 'C-Near', 'R-Near']

        for i, zone in enumerate(zones):
            row = i // 3
            col = i % 3
            x = col * width // 3 + 10
            y = row * height // 3 + 25
            cv2.putText(frame, zone, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _display_frame(self, frame, skip=False):
        """Display frame in window"""
        if not Config.ENABLE_VISUALIZATION:
            return

        title = "ThermoVision" + (" [SKIPPED]" if skip else "")
        cv2.imshow(title, frame)

    def _shutdown(self):
        """Cleanup and shutdown"""
        print("\n🔄 Shutting down ThermoVision...")

        if self.audio:
            self.audio.speak("ThermoVision stopped", priority="low")
            time.sleep(1)  # Wait for speech to finish
            self.audio.shutdown()

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        print("✅ Shutdown complete")
        print("=" * 60)


def main():
    """Entry point"""
    try:
        app = ThermoVision()
        app.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())