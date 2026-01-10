"""
Hand Gesture Detection Test
===========================
Detects two gestures:
- OPEN PALM (all fingers extended)
- CLOSED FIST (all fingers closed)

Press 'Q' to quit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os


class HandGestureDetector:
    """Simple hand gesture detector using MediaPipe Tasks API"""
    
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
        
        # Gesture states
        self.OPEN_PALM = "OPEN_PALM"
        self.CLOSED_FIST = "CLOSED_FIST"
        self.BACK_OF_HAND = "BACK_OF_HAND"
        self.UNKNOWN = "UNKNOWN"
        
        # Gesture smoothing (to avoid flickering)
        self.gesture_history = []
        self.history_size = 5  # Number of frames to average
        
        # Drawing specs
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
            # For thumb, check if tip is far from the base of index finger
            index_mcp = landmarks[5]
            return self._distance(tip, index_mcp) > self._distance(mcp, index_mcp) * 1.2
        else:
            # For other fingers: tip should be above PIP (lower y = higher on screen)
            # Also check that tip is farther from wrist than PIP
            wrist = landmarks[0]
            tip_to_wrist = self._distance(tip, wrist)
            pip_to_wrist = self._distance(pip, wrist)
            
            # Finger extended if tip is higher than PIP OR tip is farther from wrist
            return tip.y < pip.y or tip_to_wrist > pip_to_wrist * 1.1
    
    def detect_gesture(self, landmarks):
        """
        Detect if hand is open palm or closed fist
        
        Simple and robust approach:
        - Count how many fingers are extended
        - 4-5 fingers extended = OPEN PALM
        - 0-1 fingers extended = CLOSED FIST
        """
        if not landmarks:
            return self.UNKNOWN
        
        # Check each finger
        fingers_extended = []
        
        # Thumb (special case)
        thumb_extended = self._is_finger_extended(landmarks, 4, 3, 2, is_thumb=True)
        fingers_extended.append(thumb_extended)
        
        # Index finger
        index_extended = self._is_finger_extended(landmarks, 8, 6, 5)
        fingers_extended.append(index_extended)
        
        # Middle finger
        middle_extended = self._is_finger_extended(landmarks, 12, 10, 9)
        fingers_extended.append(middle_extended)
        
        # Ring finger
        ring_extended = self._is_finger_extended(landmarks, 16, 14, 13)
        fingers_extended.append(ring_extended)
        
        # Pinky finger
        pinky_extended = self._is_finger_extended(landmarks, 20, 18, 17)
        fingers_extended.append(pinky_extended)
        
        # Count extended fingers
        extended_count = sum(fingers_extended)
        
        # Store for debugging (optional)
        self.last_extended_count = extended_count
        self.last_fingers = fingers_extended
        
        # Decision: be more lenient with thresholds
        if extended_count >= 4:  # 4 or 5 fingers = OPEN
            return self.OPEN_PALM
        elif extended_count <= 1:  # 0 or 1 finger = FIST
            return self.CLOSED_FIST
        else:
            # For 2-3 fingers, use additional heuristics
            # Check if fingertips are close together (fist-like)
            tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            palm_center_y = sum(t.y for t in tips) / 4
            wrist_y = landmarks[0].y
            
            # If fingertips are closer to wrist (curled down), it's more like a fist
            if palm_center_y > wrist_y - 0.1:  # Tips below or near wrist level
                return self.CLOSED_FIST
            
            return self.UNKNOWN
    def _smooth_gesture(self, raw_gesture):
        """Smooth gesture detection to avoid flickering"""
        self.gesture_history.append(raw_gesture)
        
        # Keep only recent history
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Count occurrences
        from collections import Counter
        counts = Counter(self.gesture_history)
        
        # Return most common gesture if it appears in majority of frames
        most_common, count = counts.most_common(1)[0]
        
        # Require at least 3 out of 5 frames for stability
        if count >= 3:
            return most_common
        
        # If no clear winner, prefer previous stable state
        if len(self.gesture_history) > 1:
            return self.gesture_history[-2]
        
        return raw_gesture
    
    def is_palm_facing(self, landmarks, handedness):
        """
        Check if the palm (front) of the hand is facing the camera.
        
        Logic: Compare thumb position relative to pinky
        - Right hand palm: thumb is to the LEFT of pinky (in mirrored view)
        - Left hand palm: thumb is to the RIGHT of pinky (in mirrored view)
        
        Since the video is mirrored, we flip the logic.
        """
        if not landmarks:
            return False
        
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        
        # Get hand label ("Left" or "Right")
        hand_label = "Right"  # default
        if handedness and len(handedness) > 0:
            hand_label = handedness[0].category_name
        
        # In mirrored video:
        # - Right hand showing palm: thumb.x > pinky.x (thumb on right side)
        # - Left hand showing palm: thumb.x < pinky.x (thumb on left side)
        
        if hand_label == "Right":
            return thumb_tip.x > pinky_tip.x
        else:  # Left hand
            return thumb_tip.x < pinky_tip.x
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        if not landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        # Define hand connections (same as MediaPipe)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections
        for start, end in connections:
            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)
            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), self.connection_color, 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, self.landmark_color, -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
        
        return frame
    
    def process_frame(self, frame):
        """Process a frame and return gesture + annotated frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        result = self.detector.detect(mp_image)
        
        raw_gesture = self.UNKNOWN
        
        if result.hand_landmarks:
            for idx, landmarks in enumerate(result.hand_landmarks):
                # Get handedness for this hand
                handedness = result.handedness[idx] if result.handedness else None
                
                # Check if palm is facing camera
                palm_facing = self.is_palm_facing(landmarks, handedness)
                self.is_palm_visible = palm_facing
                
                # Draw landmarks (different color for back of hand)
                if palm_facing:
                    self.landmark_color = (0, 255, 150)  # Green for palm
                else:
                    self.landmark_color = (0, 100, 255)  # Orange for back
                
                frame = self.draw_landmarks(frame, landmarks)
                
                # Only detect gesture if palm is facing camera
                if palm_facing:
                    raw_gesture = self.detect_gesture(landmarks)
                else:
                    raw_gesture = self.BACK_OF_HAND
        
        # Apply smoothing for stable output
        gesture = self._smooth_gesture(raw_gesture)
        
        return gesture, frame
    
    def release(self):
        """Clean up resources"""
        pass


def draw_ui(frame, gesture, detector):
    """Draw beautiful UI overlay on frame"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for status bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "HAND GESTURE DETECTOR", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Gesture status with color coding
    if gesture == detector.OPEN_PALM:
        status_text = "OPEN PALM"
        status_color = (0, 255, 150)  # Green
    elif gesture == detector.CLOSED_FIST:
        status_text = "CLOSED FIST"
        status_color = (0, 150, 255)  # Orange
    elif gesture == detector.BACK_OF_HAND:
        status_text = "BACK OF HAND (show palm!)"
        status_color = (0, 100, 255)  # Orange-red
    else:
        status_text = "NO HAND / UNKNOWN"
        status_color = (150, 150, 150)  # Gray
    
    cv2.putText(frame, f"Gesture: {status_text}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Show finger count for debugging
    if hasattr(detector, 'last_extended_count'):
        finger_info = f"Fingers: {detector.last_extended_count}/5"
        cv2.putText(frame, finger_info, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw gesture indicator circle
    center_x = width - 60
    center_y = 50
    
    if gesture == detector.OPEN_PALM:
        # Draw open hand indicator (circle with rays)
        cv2.circle(frame, (center_x, center_y), 25, status_color, 3)
        for angle in range(0, 360, 72):
            rad = np.radians(angle)
            x1 = int(center_x + 25 * np.cos(rad))
            y1 = int(center_y + 25 * np.sin(rad))
            x2 = int(center_x + 35 * np.cos(rad))
            y2 = int(center_y + 35 * np.sin(rad))
            cv2.line(frame, (x1, y1), (x2, y2), status_color, 2)
    elif gesture == detector.CLOSED_FIST:
        # Draw fist indicator (filled circle)
        cv2.circle(frame, (center_x, center_y), 25, status_color, -1)
        cv2.circle(frame, (center_x, center_y), 25, (255, 255, 255), 2)
    else:
        # Draw empty indicator
        cv2.circle(frame, (center_x, center_y), 25, status_color, 2)
    
    # Instructions at bottom
    cv2.rectangle(frame, (0, height - 40), (width, height), (30, 30, 30), -1)
    cv2.putText(frame, "Show OPEN PALM or CLOSED FIST | Press 'Q' to quit",
                (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


def main():
    """Main function to run gesture detection"""
    print("\n" + "="*50)
    print("   HAND GESTURE DETECTION TEST")
    print("="*50)
    print("\nInitializing camera and MediaPipe...")
    
    # Initialize detector
    detector = HandGestureDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nCamera ready!")
    print("MediaPipe ready!")
    print("\n" + "-"*50)
    print("GESTURES TO TEST:")
    print("  OPEN PALM  - Extend all fingers")
    print("  CLOSED FIST - Close all fingers")
    print("-"*50)
    print("\nPress 'Q' to quit\n")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame")
                break
            
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Process frame
            gesture, frame = detector.process_frame(frame)
            
            # Draw UI
            frame = draw_ui(frame, gesture, detector)
            
            # Show frame
            cv2.imshow("Hand Gesture Test", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
