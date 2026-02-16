"""
ThermoVision - Fire Detection Module
Detects fire and flames using color analysis + CNN
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time


class FireCNN(nn.Module):
    """
    Lightweight CNN for fire classification
    Architecture: MobileNet-inspired for speed
    """

    def __init__(self):
        super(FireCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary: fire or no-fire
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FireDetector:
    """
    Fire and flame detection using hybrid approach:
    1. Color-based detection (fast, real-time)
    2. CNN-based classification (accurate)
    3. Motion/flicker analysis
    """

    def __init__(self, confidence_threshold=0.6, use_gpu=False):
        """
        Initialize fire detector

        Args:
            confidence_threshold: Minimum confidence for fire detection
            use_gpu: Use GPU if available
        """
        self.confidence_threshold = confidence_threshold

        # Device setup
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"🔥 Fire Detector using: {self.device}")

        # Initialize CNN model
        self.model = FireCNN().to(self.device)
        self.model.eval()  # Inference mode

        # Color thresholds (HSV)
        # Fire colors: Orange/Red/Yellow
        self.fire_color_lower = np.array([0, 100, 100])    # Lower HSV bound
        self.fire_color_upper = np.array([30, 255, 255])   # Upper HSV bound

        # Additional yellow range for flames
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([40, 255, 255])

        # Motion history for flicker detection
        self.motion_history = deque(maxlen=10)
        self.prev_frame_gray = None

        # Detection history for temporal consistency
        self.detection_history = deque(maxlen=5)

        # Alert cooldown management (3 seconds minimum)
        self.last_fire_alert_time = 0
        self.fire_alert_cooldown = 3.0  # 3 SECONDS between fire alerts

        # Performance tracking
        self.inference_times = deque(maxlen=30)

        print("✅ Fire detector initialized")

    def detect(self, frame):
        """
        Detect fire in frame using hybrid approach

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict: {
                'fire_detected': bool,
                'confidence': float,
                'bounding_boxes': list of (x, y, w, h),
                'fire_regions': list of contours,
                'method': str (color/cnn/hybrid),
                'should_alert': bool (NEW - respects 3s cooldown)
            }
        """
        start_time = time.time()

        # Step 1: Fast color-based pre-filtering
        color_result = self._detect_by_color(frame)

        # Step 2: If color detection finds candidates, verify with CNN
        if color_result['candidates'] > 0:
            # Extract fire region for CNN verification
            cnn_result = self._verify_with_cnn(frame, color_result['mask'])

            # Step 3: Motion/flicker analysis
            motion_score = self._analyze_motion(frame)

            # Combine results
            final_confidence = self._fuse_detections(
                color_result['confidence'],
                cnn_result['confidence'],
                motion_score
            )

            fire_detected = final_confidence >= self.confidence_threshold
            method = "hybrid"
        else:
            # No color candidates, likely no fire
            fire_detected = False
            final_confidence = 0.0
            method = "color"
            color_result['bounding_boxes'] = []

        # Temporal smoothing
        self.detection_history.append(fire_detected)
        fire_detected = sum(self.detection_history) >= 3  # 3 out of 5 frames

        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        # Check if we should alert (3-second cooldown)
        current_time = time.time()
        should_alert = False
        if fire_detected:
            if current_time - self.last_fire_alert_time >= self.fire_alert_cooldown:
                should_alert = True
                self.last_fire_alert_time = current_time

        result = {
            'fire_detected': fire_detected,
            'confidence': final_confidence,
            'bounding_boxes': color_result['bounding_boxes'],
            'fire_mask': color_result['mask'],
            'method': method,
            'inference_time': inference_time,
            'should_alert': should_alert  # NEW
        }

        return result

    def _detect_by_color(self, frame):
        """
        Detect fire-like colors in frame

        Args:
            frame: BGR image

        Returns:
            dict with color detection results
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for fire colors
        mask1 = cv2.inRange(hsv, self.fire_color_lower, self.fire_color_upper)
        mask2 = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Combine masks
        fire_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to reduce noise (REDUCED kernel size)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel_small)  # Remove small noise first
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel_small)  # Fill small holes

        # Find contours
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter small contours (noise) - INCREASED minimum area
        min_area = 800  # Minimum pixels for fire region (more strict)
        max_area = frame.shape[0] * frame.shape[1] * 0.3  # Maximum 30% of frame
        valid_contours = [cnt for cnt in contours
                         if min_area < cv2.contourArea(cnt) < max_area]

        # Get bounding boxes
        bounding_boxes = []
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))

        # Calculate confidence based on color match strength
        if len(valid_contours) > 0:
            # Calculate total fire pixels from valid contours only
            total_fire_pixels = sum(cv2.contourArea(cnt) for cnt in valid_contours)
            frame_size = frame.shape[0] * frame.shape[1]

            # More conservative confidence calculation
            pixel_ratio = total_fire_pixels / frame_size

            # Penalize if fire takes up too much of frame (likely false positive)
            if pixel_ratio > 0.3:  # More than 30% of frame
                confidence = pixel_ratio * 0.3  # Heavy penalty
            else:
                confidence = min(pixel_ratio * 5, 1.0)  # Scale up small fires
        else:
            confidence = 0.0

        return {
            'candidates': len(valid_contours),
            'confidence': confidence,
            'mask': fire_mask,
            'bounding_boxes': bounding_boxes
        }

    def _verify_with_cnn(self, frame, fire_mask):
        """
        Verify fire detection using CNN

        Args:
            frame: Original BGR frame
            fire_mask: Binary mask of potential fire regions

        Returns:
            dict with CNN verification results
        """
        # Extract region of interest
        roi = cv2.bitwise_and(frame, frame, mask=fire_mask)

        # Preprocess for CNN
        input_tensor = self._preprocess_for_cnn(roi)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            fire_prob = probabilities[0][1].item()  # Probability of fire class

        return {
            'confidence': fire_prob,
            'is_fire': fire_prob > 0.5
        }

    def _preprocess_for_cnn(self, frame):
        """
        Preprocess frame for CNN input

        Args:
            frame: BGR image

        Returns:
            Tensor ready for CNN
        """
        # Resize to model input size
        resized = cv2.resize(frame, (128, 128))

        # Convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def _analyze_motion(self, frame):
        """
        Analyze motion/flicker characteristic of fire

        Args:
            frame: Current frame

        Returns:
            float: Motion score (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return 0.0

        # Compute frame difference
        diff = cv2.absdiff(gray, self.prev_frame_gray)

        # Calculate motion intensity
        motion_intensity = np.mean(diff)

        # Store in history
        self.motion_history.append(motion_intensity)

        # Update previous frame
        self.prev_frame_gray = gray

        # Fire has characteristic flickering (variance in motion)
        if len(self.motion_history) >= 5:
            motion_variance = np.var(list(self.motion_history))
            # Normalize variance to 0-1 range
            motion_score = min(motion_variance / 100.0, 1.0)
        else:
            motion_score = 0.0

        return motion_score

    def _fuse_detections(self, color_conf, cnn_conf, motion_score):
        """
        Fuse multiple detection methods

        Args:
            color_conf: Color-based confidence
            cnn_conf: CNN confidence
            motion_score: Motion analysis score

        Returns:
            float: Final fused confidence
        """
        # Weighted fusion
        weights = {
            'color': 0.3,
            'cnn': 0.5,
            'motion': 0.2
        }

        fused = (
            weights['color'] * color_conf +
            weights['cnn'] * cnn_conf +
            weights['motion'] * motion_score
        )

        return fused

    def get_average_inference_time(self):
        """Get average inference time in milliseconds"""
        if len(self.inference_times) == 0:
            return 0.0
        return np.mean(list(self.inference_times)) * 1000

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

        if detection_result['fire_detected']:
            # Draw bounding boxes ONLY (not full mask overlay)
            for (x, y, w, h) in detection_result['bounding_boxes']:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 100, 255), 3)

                # Add label
                label = f"FIRE {detection_result['confidence']:.2f}"
                cv2.putText(annotated, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            # Add warning text
            cv2.putText(annotated, "!!! FIRE DETECTED !!!", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # REMOVED: Full mask overlay that was causing entire screen to turn red
        # Only show bounding boxes now

        return annotated


# Standalone test function
def test_fire_detector():
    """
    Test fire detector with webcam
    Also tests with sample fire images/videos if available
    """
    print("=" * 60)
    print("Fire Detector - Standalone Test")
    print("=" * 60)
    print("Controls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("=" * 60)
    print()

    # Initialize detector
    detector = FireDetector(confidence_threshold=0.5)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("📹 Camera opened. Show fire/flames to test detection...")
    print()

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect fire
            result = detector.detect(frame)

            # Visualize
            annotated = detector.visualize_detection(frame, result)

            # Add info overlay
            avg_time = detector.get_average_inference_time()
            info_text = f"Inference: {avg_time:.1f}ms | Method: {result['method']}"
            cv2.putText(annotated, info_text, (10, annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show result
            cv2.imshow("Fire Detector Test", annotated)

            # Print detection
            if result['fire_detected'] and frame_count % 30 == 0:
                print(f"🔥 FIRE DETECTED! Confidence: {result['confidence']:.2%}")

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"fire_detection_{frame_count}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"💾 Saved: {filename}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test complete")
        print(f"Average inference time: {detector.get_average_inference_time():.1f}ms")


if __name__ == "__main__":
    test_fire_detector()