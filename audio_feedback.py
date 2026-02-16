"""
ThermoVision - Audio Feedback Module
Provides text-to-speech alerts for hazard warnings
"""

import pyttsx3
import time
from collections import defaultdict
import threading


class AudioFeedback:
    """
    Manages audio alerts using text-to-speech
    Includes cooldown to prevent alert spam
    """

    def __init__(self, rate=150, volume=1.0, cooldown=3.0):
        """
        Initialize audio feedback system

        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            cooldown: Minimum seconds between same alerts
        """
        self.rate = rate
        self.volume = volume
        self.cooldown = cooldown

        # Initialize TTS engine
        self.engine = None
        self._initialize_engine()

        # Alert tracking for cooldown
        self.last_alert_time = defaultdict(float)

        # Thread safety for async speaking
        self.lock = threading.Lock()
        self.is_speaking = False

    def _initialize_engine(self):
        """Initialize pyttsx3 engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)

            # Try to set a clear voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice if available (index 1 usually)
                if len(voices) > 1:
                    self.engine.setProperty('voice', voices[1].id)
                else:
                    self.engine.setProperty('voice', voices[0].id)

            print("🔊 Audio feedback initialized")

        except Exception as e:
            print(f"⚠️ Audio initialization warning: {e}")
            self.engine = None

    def speak(self, text, priority="normal"):
        """
        Speak text with cooldown management

        Args:
            text: Message to speak
            priority: "low", "normal", "high", "critical"
                     Critical alerts bypass cooldown

        Returns:
            bool: True if spoken, False if on cooldown
        """
        if not self.engine:
            print(f"[AUDIO DISABLED] {text}")
            return False

        current_time = time.time()

        # Check cooldown (except for critical alerts)
        if priority != "critical":
            if current_time - self.last_alert_time[text] < self.cooldown:
                return False  # Still on cooldown

        # Update last alert time
        self.last_alert_time[text] = current_time

        # Speak in separate thread to avoid blocking
        thread = threading.Thread(target=self._speak_async, args=(text,))
        thread.daemon = True
        thread.start()

        return True

    def _speak_async(self, text):
        """
        Internal method to speak asynchronously

        Args:
            text: Message to speak
        """
        with self.lock:
            if self.is_speaking:
                return  # Already speaking

            self.is_speaking = True

            try:
                print(f"🔊 Speaking: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"⚠️ Speech error: {e}")
            finally:
                self.is_speaking = False

    def alert_fire(self, direction="ahead", distance="close"):
        """
        Fire alert

        Args:
            direction: "left", "right", "ahead", "behind"
            distance: "close", "medium", "far"
        """
        urgency = "critical" if distance == "close" else "high"
        message = f"Fire detected {direction}"

        if distance == "close":
            message = f"Warning! {message}. Move away immediately"

        self.speak(message, priority=urgency)

    def alert_human(self, direction="ahead", distance="close"):
        """
        Human presence alert

        Args:
            direction: "left", "right", "ahead", "behind"
            distance: "close", "medium", "far"
        """
        if distance == "close":
            message = f"Person very close on your {direction}"
            priority = "high"
        else:
            message = f"Person detected {direction}"
            priority = "normal"

        self.speak(message, priority=priority)

    def alert_hot_object(self, object_type, direction="ahead"):
        """
        Hot object alert

        Args:
            object_type: Type of object (e.g., "car", "stove")
            direction: "left", "right", "ahead", "behind"
        """
        message = f"Hot {object_type} {direction}"
        self.speak(message, priority="high")

    def alert_danger_zone(self):
        """Critical danger alert"""
        self.speak("Danger! Stop immediately", priority="critical")

    def alert_clear(self):
        """All clear notification"""
        self.speak("Path clear", priority="low")

    def alert_custom(self, message, priority="normal"):
        """
        Custom alert message

        Args:
            message: Custom message to speak
            priority: Alert priority level
        """
        self.speak(message, priority=priority)

    def set_rate(self, rate):
        """Change speech rate"""
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """Change volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.volume)

    def shutdown(self):
        """Cleanup audio engine"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        print("🔇 Audio feedback shutdown")


# Test function
def test_audio():
    """Test audio feedback system"""
    print("=" * 60)
    print("ThermoVision - Audio Feedback Test")
    print("=" * 60)
    print()

    audio = AudioFeedback(rate=150, cooldown=2.0)

    print("Testing various alerts...")
    print()

    # Test alerts
    audio.alert_fire(direction="ahead", distance="close")
    time.sleep(3)

    audio.alert_human(direction="left", distance="medium")
    time.sleep(3)

    audio.alert_hot_object("car", direction="right")
    time.sleep(3)

    audio.alert_danger_zone()
    time.sleep(3)

    audio.alert_clear()
    time.sleep(2)

    print()
    print("Testing cooldown (should skip duplicate)...")
    audio.speak("Test message", priority="normal")
    time.sleep(0.5)
    audio.speak("Test message", priority="normal")  # Should be skipped
    time.sleep(3)

    print()
    print("✅ Audio test complete")
    audio.shutdown()


if __name__ == "__main__":
    test_audio()