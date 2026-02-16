"""
ThermoVision - Low-Light Enhancement Module
Enables detection in dark/low-light environments
KEY NOVELTY FEATURE
"""

import cv2
import numpy as np
from collections import deque


class LowLightEnhancer:
    """
    Enhance low-light images for better detection
    Simulates night vision/thermal imaging using only RGB camera
    """

    def __init__(self, enable_auto=True, enhancement_level=2.0):
        """
        Initialize low-light enhancer

        Args:
            enable_auto: Auto-detect low light and enhance
            enhancement_level: Enhancement strength (1.0-3.0)
        """
        self.enable_auto = enable_auto
        self.enhancement_level = enhancement_level

        # Brightness history for adaptive adjustment
        self.brightness_history = deque(maxlen=30)

        # Thresholds
        self.low_light_threshold = 80  # Mean brightness below this = low light
        self.very_dark_threshold = 40  # Very dark conditions

        # Current mode
        self.is_low_light = False
        self.enhancement_mode = "normal"  # normal, low_light, very_dark

        print("🌙 Low-Light Enhancer initialized")
        print(f"   Auto-detection: {'Enabled' if enable_auto else 'Disabled'}")
        print(f"   Enhancement level: {enhancement_level}x")

    def detect_lighting_conditions(self, frame):
        """
        Detect current lighting conditions

        Args:
            frame: Input BGR frame

        Returns:
            tuple: (is_low_light, mean_brightness, mode)
        """
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate mean brightness
        mean_brightness = np.mean(gray)

        # Add to history
        self.brightness_history.append(mean_brightness)

        # Smooth brightness using history
        avg_brightness = np.mean(list(self.brightness_history))

        # Determine mode
        if avg_brightness < self.very_dark_threshold:
            mode = "very_dark"
            is_low_light = True
        elif avg_brightness < self.low_light_threshold:
            mode = "low_light"
            is_low_light = True
        else:
            mode = "normal"
            is_low_light = False

        return is_low_light, avg_brightness, mode

    def enhance(self, frame):
        """
        Main enhancement pipeline

        Args:
            frame: Input BGR frame

        Returns:
            tuple: (enhanced_frame, metadata)
        """
        # Detect lighting
        is_low_light, brightness, mode = self.detect_lighting_conditions(frame)

        self.is_low_light = is_low_light
        self.enhancement_mode = mode

        # If normal lighting, return as-is
        if not is_low_light or not self.enable_auto:
            return frame, {
                'enhanced': False,
                'brightness': brightness,
                'mode': mode
            }

        # Apply enhancement based on mode
        if mode == "very_dark":
            enhanced = self._enhance_very_dark(frame)
        else:  # low_light
            enhanced = self._enhance_low_light(frame)

        return enhanced, {
            'enhanced': True,
            'brightness': brightness,
            'mode': mode,
            'original_brightness': brightness,
            'enhanced_brightness': np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        }

    def _enhance_low_light(self, frame):
        """
        Enhance low-light image (moderate darkness)

        Args:
            frame: Input frame

        Returns:
            Enhanced frame
        """
        # Method 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Method 2: Gamma correction
        gamma = 1.5  # Brighten
        enhanced = self._adjust_gamma(enhanced, gamma)

        # Method 3: Bilateral filter (denoise while preserving edges)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return enhanced

    def _enhance_very_dark(self, frame):
        """
        Enhance very dark image (extreme low light)

        Args:
            frame: Input frame

        Returns:
            Enhanced frame
        """
        # Aggressive enhancement for very dark conditions

        # Step 1: Strong CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Step 2: Strong gamma correction
        gamma = 2.0  # More aggressive brightening
        enhanced = self._adjust_gamma(enhanced, gamma)

        # Step 3: Increase saturation for better color detection
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Boost saturation
        s = cv2.multiply(s, 1.5)
        s = np.clip(s, 0, 255).astype(np.uint8)

        # Boost value (brightness)
        v = cv2.add(v, 30)
        v = np.clip(v, 0, 255).astype(np.uint8)

        enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)

        # Step 4: Noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 100, 100)

        return enhanced

    def _adjust_gamma(self, frame, gamma=1.0):
        """
        Adjust image gamma (brightness)

        Args:
            frame: Input frame
            gamma: Gamma value (>1 = brighter, <1 = darker)

        Returns:
            Gamma-corrected frame
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # Apply gamma correction
        return cv2.LUT(frame, table)

    def create_thermal_visualization(self, frame):
        """
        Create thermal camera-like visualization (DEMO MODE)
        Simulates infrared/thermal imaging

        Args:
            frame: Input frame

        Returns:
            Thermal-style visualization
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply thermal colormap (COLORMAP_JET or COLORMAP_HOT)
        thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)

        # Boost brightness in thermal mode
        thermal = cv2.convertScaleAbs(thermal, alpha=1.2, beta=20)

        return thermal

    def get_adaptive_thresholds(self):
        """
        Get adjusted detection thresholds for current lighting

        Returns:
            dict: Adjusted thresholds for detectors
        """
        if self.enhancement_mode == "very_dark":
            return {
                'fire_confidence': 0.40,  # Lower threshold in dark
                'fire_min_area': 300,  # Smaller areas acceptable
                'human_confidence': 0.40,  # Lower threshold
                'vehicle_confidence': 0.35  # Lower threshold
            }
        elif self.enhancement_mode == "low_light":
            return {
                'fire_confidence': 0.45,
                'fire_min_area': 350,
                'human_confidence': 0.45,
                'vehicle_confidence': 0.40
            }
        else:  # normal
            return {
                'fire_confidence': 0.50,
                'fire_min_area': 400,
                'human_confidence': 0.50,
                'vehicle_confidence': 0.45
            }

    def visualize_info(self, frame, metadata):
        """
        Add low-light info overlay

        Args:
            frame: Frame to annotate
            metadata: Enhancement metadata

        Returns:
            Annotated frame
        """
        display = frame.copy()

        # Low-light indicator
        if metadata.get('enhanced', False):
            mode = metadata['mode']
            brightness = metadata['brightness']

            # Mode indicator
            if mode == "very_dark":
                color = (0, 0, 255)  # Red
                text = "🌙 NIGHT MODE"
            else:
                color = (0, 165, 255)  # Orange
                text = "🌙 LOW LIGHT"

            cv2.putText(display, text, (10, display.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Brightness info
            cv2.putText(display, f"Brightness: {brightness:.0f}",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display


# Standalone test
def test_low_light_enhancer():
    """Test low-light enhancement with webcam"""
    print("=" * 60)
    print("Low-Light Enhancement Test")
    print("=" * 60)
    print("Controls:")
    print("  Q - Quit")
    print("  T - Toggle thermal visualization")
    print("  E - Toggle enhancement")
    print("  +/- - Adjust enhancement level")
    print("=" * 60)
    print()

    enhancer = LowLightEnhancer(enable_auto=True, enhancement_level=2.0)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    show_thermal = False

    print("🌙 Testing low-light enhancement...")
    print("💡 Cover camera or go to dark room to test")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance frame
            enhanced, metadata = enhancer.enhance(frame)

            # Show thermal if enabled
            if show_thermal:
                thermal = enhancer.create_thermal_visualization(enhanced)
                display = thermal
                mode_text = "THERMAL MODE"
            else:
                display = enhanced
                mode_text = metadata['mode'].upper()

            # Add info
            display = enhancer.visualize_info(display, metadata)

            # Add mode indicator
            cv2.putText(display, mode_text, (display.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show comparison (side-by-side if enhanced)
            if metadata['enhanced']:
                # Resize for side-by-side
                h, w = frame.shape[:2]
                small_original = cv2.resize(frame, (w // 3, h // 3))

                # Add "Original" label
                cv2.putText(small_original, "Original", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Place in corner
                display[10:10 + h // 3, 10:10 + w // 3] = small_original

            cv2.imshow("Low-Light Enhancement Test", display)

            # Print status
            if metadata['enhanced']:
                print(f"🌙 {metadata['mode'].upper()}: "
                      f"Brightness {metadata['original_brightness']:.0f} → "
                      f"{metadata['enhanced_brightness']:.0f}")

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                show_thermal = not show_thermal
                print(f"Thermal mode: {'ON' if show_thermal else 'OFF'}")
            elif key == ord('e'):
                enhancer.enable_auto = not enhancer.enable_auto
                print(f"Enhancement: {'ON' if enhancer.enable_auto else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                enhancer.enhancement_level = min(enhancer.enhancement_level + 0.5, 3.0)
                print(f"Enhancement level: {enhancer.enhancement_level}")
            elif key == ord('-'):
                enhancer.enhancement_level = max(enhancer.enhancement_level - 0.5, 1.0)
                print(f"Enhancement level: {enhancer.enhancement_level}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test complete")


if __name__ == "__main__":
    test_low_light_enhancer()