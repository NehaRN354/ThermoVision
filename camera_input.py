"""
ThermoVision - Camera Input Module
Handles webcam/phone camera capture with stabilization for neck-worn setup
"""

import cv2
import numpy as np
from collections import deque
import time


class CameraInput:
    """
    Manages camera feed capture with motion stabilization
    Optimized for neck-worn/lanyard camera setup
    """

    def __init__(self, camera_id=0, resolution=(640, 480), fps_target=30):
        """
        Initialize camera input

        Args:
            camera_id: Camera device ID (0 for default webcam)
            resolution: Tuple (width, height)
            fps_target: Target frames per second
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps_target = fps_target

        # Camera object
        self.cap = None
        self.is_running = False

        # Frame stabilization
        self.frame_buffer = self.frame_buffer = deque(maxlen=5) # Store last 3 frames for smoothing
        self.prev_frame = None

        # Performance metrics
        self.frame_count = 0
        self.start_time = None
        self.current_fps = 0

        # Initialize camera
        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize and configure the camera"""
        print(f"🎥 Initializing camera source: {self.camera_id}...")

        # -----------------------------------
        # Detect if source is IP camera
        # -----------------------------------
        if isinstance(self.camera_id, str) and self.camera_id.startswith("http"):
            # IP Camera (DroidCam WiFi)
            self.cap = cv2.VideoCapture(self.camera_id)

        else:
            # Local camera (webcam / USB DroidCam)

            # Try DirectShow first (FAST on Windows)
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

            # Fallback if it fails
            if not self.cap.isOpened():
                print("⚠️ DirectShow failed, trying MSMF backend...")
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)

        # -----------------------------------
        # Validate
        # -----------------------------------
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open camera source: {self.camera_id}")

        # -----------------------------------
        # Set properties (ONLY for local cams)
        # IP cameras ignore these anyway
        # -----------------------------------
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"✅ Camera initialized: {actual_width}x{actual_height}")

        self.is_running = True
        self.start_time = time.time()

    def read_frame(self):
        """
        Read a single frame from camera

        Returns:
            tuple: (success, frame) - success is bool, frame is numpy array
        """
        if not self.is_running:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            print("❌ Failed to read frame")
            return False, None

        # Update FPS counter
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.current_fps = self.frame_count / elapsed

        return True, frame

    def read_stabilized_frame(self):
        """
        Read frame with motion stabilization
        Uses temporal smoothing to reduce neck-worn camera sway

        Returns:
            tuple: (success, stabilized_frame)
        """
        ret, frame = self.read_frame()

        if not ret:
            return False, None

        # Add to buffer
        self.frame_buffer.append(frame)

        # Need at least 2 frames for stabilization
        if len(self.frame_buffer) < 2:
            return True, frame

        # Simple temporal averaging for stabilization
        stabilized = self._temporal_smooth(list(self.frame_buffer))

        return True, stabilized

    def _temporal_smooth(self, frames):
        """
        Apply temporal smoothing to reduce motion blur

        Args:
            frames: List of frames to smooth

        Returns:
            Smoothed frame
        """
        if len(frames) == 1:
            return frames[0]

        # Weighted average: newer frames have more weight
        weights = np.array([0.2, 0.3, 0.5])[:len(frames)]
        weights = weights / weights.sum()

        smoothed = np.zeros_like(frames[0], dtype=np.float32)

        for frame, weight in zip(frames, weights):
            smoothed += frame.astype(np.float32) * weight

        return smoothed.astype(np.uint8)

    def get_frame_for_detection(self):
        """
        Get optimized frame for AI detection
        Includes preprocessing and normalization

        Returns:
            tuple: (success, processed_frame, original_frame)
        """
        ret, frame = self.read_stabilized_frame()

        if not ret:
            return False, None, None

        # Keep original for display
        original = frame.copy()

        # Preprocessing for AI models
        processed = self._preprocess_for_detection(frame)

        return True, processed, original

    def _preprocess_for_detection(self, frame):
        """
        Preprocess frame for AI detection

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        # Enhance contrast for better detection
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def get_fps(self):
        """Get current FPS"""
        return round(self.current_fps, 1)

    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print("📷 Camera released")

    def __del__(self):
        """Destructor to ensure camera is released"""
        self.release()


# Quick test function
def test_camera():
    """Test camera input with visualization"""
    print("=" * 60)
    print("ThermoVision - Camera Input Test")
    print("=" * 60)
    print("Press 'q' to quit, 's' to toggle stabilization")
    print()

    camera = CameraInput(camera_id=0, resolution=(416,416))
    use_stabilization = True

    try:
        while True:
            if use_stabilization:
                ret, frame = camera.read_stabilized_frame()
                mode = "STABILIZED"
            else:
                ret, frame = camera.read_frame()
                mode = "RAW"

            if not ret:
                break

            # Add info overlay
            fps = camera.get_fps()
            cv2.putText(frame, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {mode}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("ThermoVision - Camera Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                use_stabilization = not use_stabilization
                print(f"Stabilization: {'ON' if use_stabilization else 'OFF'}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()