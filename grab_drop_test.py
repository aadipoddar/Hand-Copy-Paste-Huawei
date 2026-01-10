"""
GRAB & DROP Gesture Detection Test
===================================
Detects gesture TRANSITIONS:
- GRAB: Open Palm → Closed Fist (like grabbing something)
- DROP: Closed Fist → Open Palm (like releasing something)

Press 'Q' to quit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time


class GrabDropDetector:
    """Detects GRAB and DROP gestures based on hand state transitions"""
    
    def __init__(self):
        # Download model if not exists
        self.model_path = self._ensure_model()
        
        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Hand states
        self.OPEN_PALM = "OPEN_PALM"
        self.CLOSED_FIST = "CLOSED_FIST"
        self.BACK_OF_HAND = "BACK_OF_HAND"
        self.UNKNOWN = "UNKNOWN"
        
        # Gesture events
        self.GRAB = "GRAB"
        self.DROP = "DROP"
        self.NONE = "NONE"
        
        # State tracking
        self.current_state = self.UNKNOWN
        self.previous_state = self.UNKNOWN
        self.confirmed_state = self.UNKNOWN  # State must be stable to be confirmed
        self.state_history = []
        self.history_size = 3  # Reduced for faster response
        
        # Timing for state confirmation
        self.state_start_time = 0
        self.state_confirm_duration = 0.1  # Only 100ms to confirm
        self.last_hand_seen_time = 0
        self.hand_timeout = 0.5  # Reset if no hand for 0.5 second
        
        # Grab/Drop state
        self.is_holding = False  # True after GRAB, False after DROP
        self.last_event = self.NONE
        self.last_event_time = 0
        self.event_display_duration = 1.0  # Show event for 1 second
        self.grab_ready = False  # Must see open palm first
        self.drop_ready = False  # Must see closed fist first
        
        # Debug info
        self.last_extended_count = 0
        self.is_palm_visible = True
        
        # Drawing
        self.landmark_color = (0, 255, 150)
        self.connection_color = (255, 255, 255)
    
    def _ensure_model(self):
        """Download the hand landmarker model if not present"""
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")
        return model_path
    
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _is_finger_extended(self, landmarks, tip_idx, pip_idx, mcp_idx, is_thumb=False):
        """Check if a single finger is extended"""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        
        if is_thumb:
            index_mcp = landmarks[5]
            return self._distance(tip, index_mcp) > self._distance(mcp, index_mcp) * 1.2
        else:
            wrist = landmarks[0]
            tip_to_wrist = self._distance(tip, wrist)
            pip_to_wrist = self._distance(pip, wrist)
            return tip.y < pip.y or tip_to_wrist > pip_to_wrist * 1.1
    
    def is_palm_facing(self, landmarks, handedness):
        """Check if the palm is facing the camera"""
        if not landmarks:
            return False
        
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        
        hand_label = "Right"
        if handedness and len(handedness) > 0:
            hand_label = handedness[0].category_name
        
        if hand_label == "Right":
            return thumb_tip.x > pinky_tip.x
        else:
            return thumb_tip.x < pinky_tip.x
    
    def detect_hand_state(self, landmarks):
        """Detect current hand state (open/closed)"""
        if not landmarks:
            return self.UNKNOWN
        
        fingers_extended = []
        
        # Check each finger
        fingers_extended.append(self._is_finger_extended(landmarks, 4, 3, 2, is_thumb=True))
        fingers_extended.append(self._is_finger_extended(landmarks, 8, 6, 5))
        fingers_extended.append(self._is_finger_extended(landmarks, 12, 10, 9))
        fingers_extended.append(self._is_finger_extended(landmarks, 16, 14, 13))
        fingers_extended.append(self._is_finger_extended(landmarks, 20, 18, 17))
        
        extended_count = sum(fingers_extended)
        self.last_extended_count = extended_count
        
        if extended_count >= 4:
            return self.OPEN_PALM
        elif extended_count <= 1:
            return self.CLOSED_FIST
        else:
            # For ambiguous states, lean towards previous state
            tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            palm_center_y = sum(t.y for t in tips) / 4
            wrist_y = landmarks[0].y
            
            if palm_center_y > wrist_y - 0.1:
                return self.CLOSED_FIST
            return self.UNKNOWN
    
    def _smooth_state(self, raw_state):
        """Minimal smoothing for fast response"""
        self.state_history.append(raw_state)
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)
        
        # Just need 2 out of 3 frames to agree
        from collections import Counter
        counts = Counter(self.state_history)
        most_common, count = counts.most_common(1)[0]
        
        if count >= 2:
            return most_common
        return raw_state
    
    def detect_grab_drop(self, current_state, hand_visible):
        """Fast GRAB/DROP detection based on state transitions"""
        event = self.NONE
        current_time = time.time()
        
        # Reset if hand not visible
        if not hand_visible:
            if current_time - self.last_hand_seen_time > self.hand_timeout:
                self.grab_ready = False
                self.drop_ready = False
                self.confirmed_state = self.UNKNOWN
            return event
        
        self.last_hand_seen_time = current_time
        
        # Fast detection - act on state changes immediately
        if current_state in [self.OPEN_PALM, self.CLOSED_FIST]:
            
            # Track ready states
            if current_state == self.OPEN_PALM:
                self.grab_ready = True
                
                # DROP: Was holding + had fist confirmed + now open
                if self.is_holding and self.confirmed_state == self.CLOSED_FIST:
                    event = self.DROP
                    self.is_holding = False
                    self.drop_ready = False
                    self.last_event = self.DROP
                    self.last_event_time = current_time
                    print(">>> DROP! <<<")
                    
            elif current_state == self.CLOSED_FIST:
                self.drop_ready = True
                
                # GRAB: Not holding + had palm confirmed + now fist
                if not self.is_holding and self.confirmed_state == self.OPEN_PALM:
                    event = self.GRAB
                    self.is_holding = True
                    self.grab_ready = False
                    self.last_event = self.GRAB
                    self.last_event_time = current_time
                    print(">>> GRAB! <<<")
            
            # Update confirmed state
            self.confirmed_state = current_state
        
        self.previous_state = current_state
        return event
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        if not landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start, end in connections:
            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)
            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), self.connection_color, 2)
        
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, self.landmark_color, -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
        
        return frame
    
    def process_frame(self, frame):
        """Process frame and detect grab/drop gestures"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        
        raw_state = self.UNKNOWN
        event = self.NONE
        
        if result.hand_landmarks:
            for idx, landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[idx] if result.handedness else None
                palm_facing = self.is_palm_facing(landmarks, handedness)
                self.is_palm_visible = palm_facing
                
                if palm_facing:
                    self.landmark_color = (0, 255, 150)
                else:
                    self.landmark_color = (0, 100, 255)
                
                frame = self.draw_landmarks(frame, landmarks)
                
                if palm_facing:
                    raw_state = self.detect_hand_state(landmarks)
        
        # Smooth the state
        current_state = self._smooth_state(raw_state)
        self.current_state = current_state
        
        # Check if hand is actually visible
        hand_visible = raw_state in [self.OPEN_PALM, self.CLOSED_FIST]
        
        # Detect grab/drop transitions (with hand visibility check)
        event = self.detect_grab_drop(current_state, hand_visible)
        
        return current_state, event, frame
    
    def release(self):
        pass


def draw_ui(frame, state, event, detector):
    """Draw UI with grab/drop status"""
    height, width = frame.shape[:2]
    
    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 120), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Title
    cv2.putText(frame, "GRAB & DROP DETECTOR", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Current state
    if state == detector.OPEN_PALM:
        state_text = "OPEN PALM"
        state_color = (0, 255, 150)
    elif state == detector.CLOSED_FIST:
        state_text = "CLOSED FIST"
        state_color = (0, 150, 255)
    elif state == detector.BACK_OF_HAND:
        state_text = "BACK OF HAND"
        state_color = (0, 100, 255)
    else:
        state_text = "NO HAND"
        state_color = (150, 150, 150)
    
    cv2.putText(frame, f"Hand: {state_text}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
    
    # Holding status with ready indicators
    if detector.is_holding:
        holding_text = "HOLDING CONTENT"
        holding_color = (0, 200, 255)  # Yellow-orange
    else:
        holding_text = "EMPTY"
        holding_color = (150, 150, 150)
    
    cv2.putText(frame, f"Status: {holding_text}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, holding_color, 2)
    
    # Ready state indicator
    if not detector.is_holding and detector.grab_ready:
        ready_text = "[Ready to GRAB]"
        ready_color = (0, 255, 255)
    elif detector.is_holding and detector.drop_ready:
        ready_text = "[Ready to DROP]"
        ready_color = (0, 255, 150)
    else:
        ready_text = ""
        ready_color = (150, 150, 150)
    
    if ready_text:
        cv2.putText(frame, ready_text, (200, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ready_color, 1)
    
    # Finger count
    cv2.putText(frame, f"Fingers: {detector.last_extended_count}/5", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Show GRAB/DROP event (with animation)
    time_since_event = time.time() - detector.last_event_time
    if time_since_event < detector.event_display_duration:
        # Calculate animation progress (0 to 1)
        progress = time_since_event / detector.event_display_duration
        
        # Event banner in center
        if detector.last_event == detector.GRAB:
            event_text = "GRAB!"
            event_color = (0, 200, 255)  # Orange
            icon = ">>>"
        else:
            event_text = "DROP!"
            event_color = (0, 255, 150)  # Green
            icon = "<<<"
        
        # Fade out effect
        alpha = 1.0 - progress
        
        # Draw event banner
        banner_y = height // 2 - 30
        
        # Semi-transparent background
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (width//4, banner_y - 20), (3*width//4, banner_y + 50), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay2, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        
        # Event text
        font_scale = 1.5 - progress * 0.5  # Shrink slightly
        text_size = cv2.getTextSize(event_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = (width - text_size[0]) // 2
        
        # Adjust color with alpha
        adj_color = tuple(int(c * alpha) for c in event_color)
        cv2.putText(frame, f"{icon} {event_text} {icon[::-1]}", (text_x - 40, banner_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, adj_color, 3)
    
    # Draw holding indicator (right side)
    indicator_x = width - 80
    indicator_y = 60
    
    if detector.is_holding:
        # Draw "holding" icon - filled box
        cv2.rectangle(frame, (indicator_x - 25, indicator_y - 25), 
                     (indicator_x + 25, indicator_y + 25), (0, 200, 255), -1)
        cv2.rectangle(frame, (indicator_x - 25, indicator_y - 25), 
                     (indicator_x + 25, indicator_y + 25), (255, 255, 255), 2)
        cv2.putText(frame, "HELD", (indicator_x - 20, indicator_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    else:
        # Draw empty box
        cv2.rectangle(frame, (indicator_x - 25, indicator_y - 25), 
                     (indicator_x + 25, indicator_y + 25), (100, 100, 100), 2)
        cv2.putText(frame, "EMPTY", (indicator_x - 22, indicator_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    
    # Instructions at bottom
    cv2.rectangle(frame, (0, height - 50), (width, height), (30, 30, 30), -1)
    cv2.putText(frame, "GRAB: Open palm -> Close fist | DROP: Close fist -> Open palm",
                (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "Press 'Q' to quit | 'R' to reset",
                (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return frame


def main():
    print("\n" + "="*50)
    print("   GRAB & DROP GESTURE TEST")
    print("="*50)
    print("\nInitializing...")
    
    detector = GrabDropDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nCamera ready!")
    print("\n" + "-"*50)
    print("HOW TO USE:")
    print("  1. Show OPEN PALM to camera")
    print("  2. CLOSE your fist -> GRAB!")
    print("  3. OPEN your palm -> DROP!")
    print("-"*50)
    print("\nPress 'Q' to quit, 'R' to reset\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            state, event, frame = detector.process_frame(frame)
            frame = draw_ui(frame, state, event, detector)
            
            cv2.imshow("Grab & Drop Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset state
                detector.is_holding = False
                detector.last_event = detector.NONE
                print("State reset!")
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print("Done!")


if __name__ == "__main__":
    main()
