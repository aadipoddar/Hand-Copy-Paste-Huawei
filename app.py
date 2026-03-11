"""
Hand Copy Paste - Modern UI Version
====================================
Beautiful desktop app for gesture-based content transfer
"""

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import sys
import time
import base64
import threading
import qrcode
import pyperclip
import subprocess
import json
import zlib
from io import BytesIO
from datetime import datetime
from tkinter import filedialog

# Import Firebase service
from firebase_service import FirebaseService


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class HandGestureDetector:
    """Detects hand gestures for grab and drop"""
    
    def __init__(self):
        self.model_path = self._ensure_model()
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # States
        self.OPEN_PALM = "OPEN_PALM"
        self.CLOSED_FIST = "CLOSED_FIST"
        self.UNKNOWN = "UNKNOWN"
        
        # Events
        self.GRAB = "GRAB"
        self.DROP = "DROP"
        self.NONE = "NONE"
        
        # Tracking
        self.state_history = []
        self.history_size = 3
        self.confirmed_state = self.UNKNOWN
        self.previous_state = self.UNKNOWN
        self.last_hand_seen_time = 0
        self.hand_timeout = 0.5
        
        # Gesture state
        self.is_holding = False
        self.grab_ready = False
        self.drop_ready = False
        self.last_event = self.NONE
        self.last_event_time = 0
        self.finger_count = 0
    
    def _ensure_model(self):
        model_path = get_resource_path("hand_landmarker.task")
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        return model_path
    
    def _distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _is_finger_extended(self, landmarks, tip_idx, pip_idx, mcp_idx, is_thumb=False):
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        
        if is_thumb:
            index_mcp = landmarks[5]
            return self._distance(tip, index_mcp) > self._distance(mcp, index_mcp) * 1.2
        else:
            wrist = landmarks[0]
            return tip.y < pip.y or self._distance(tip, wrist) > self._distance(pip, wrist) * 1.1
    
    def is_palm_facing(self, landmarks, handedness):
        if not landmarks:
            return False
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        hand_label = handedness[0].category_name if handedness else "Right"
        
        if hand_label == "Right":
            return thumb_tip.x > pinky_tip.x
        return thumb_tip.x < pinky_tip.x
    
    def detect_hand_state(self, landmarks):
        if not landmarks:
            return self.UNKNOWN
        
        extended = [
            self._is_finger_extended(landmarks, 4, 3, 2, is_thumb=True),
            self._is_finger_extended(landmarks, 8, 6, 5),
            self._is_finger_extended(landmarks, 12, 10, 9),
            self._is_finger_extended(landmarks, 16, 14, 13),
            self._is_finger_extended(landmarks, 20, 18, 17)
        ]
        
        count = sum(extended)
        self.finger_count = count
        
        if count >= 4:
            return self.OPEN_PALM
        elif count <= 1:
            return self.CLOSED_FIST
        return self.UNKNOWN
    
    def _smooth_state(self, raw_state):
        self.state_history.append(raw_state)
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)
        
        from collections import Counter
        counts = Counter(self.state_history)
        most_common, count = counts.most_common(1)[0]
        return most_common if count >= 2 else raw_state
    
    def detect_gesture(self, current_state, hand_visible):
        event = self.NONE
        current_time = time.time()
        
        if not hand_visible:
            if current_time - self.last_hand_seen_time > self.hand_timeout:
                self.grab_ready = False
                self.drop_ready = False
                self.confirmed_state = self.UNKNOWN
            return event
        
        self.last_hand_seen_time = current_time
        
        if current_state in [self.OPEN_PALM, self.CLOSED_FIST]:
            if current_state == self.OPEN_PALM:
                self.grab_ready = True
                if self.is_holding and self.confirmed_state == self.CLOSED_FIST:
                    event = self.DROP
                    self.is_holding = False
                    self.drop_ready = False
                    self.last_event = self.DROP
                    self.last_event_time = current_time
                    
            elif current_state == self.CLOSED_FIST:
                self.drop_ready = True
                if not self.is_holding and self.confirmed_state == self.OPEN_PALM:
                    event = self.GRAB
                    self.is_holding = True
                    self.grab_ready = False
                    self.last_event = self.GRAB
                    self.last_event_time = current_time
            
            self.confirmed_state = current_state
        
        self.previous_state = current_state
        return event
    
    def draw_landmarks(self, frame, landmarks):
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
        
        # Draw connections
        for start, end in connections:
            x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Draw landmarks with gradient color
        for i, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 150) if self.is_holding else (100, 200, 255)
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        raw_state = self.UNKNOWN
        event = self.NONE
        palm_facing = False
        
        if result.hand_landmarks:
            for idx, landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[idx] if result.handedness else None
                palm_facing = self.is_palm_facing(landmarks, handedness)
                
                if palm_facing:
                    frame = self.draw_landmarks(frame, landmarks)
                    raw_state = self.detect_hand_state(landmarks)
        
        current_state = self._smooth_state(raw_state)
        hand_visible = raw_state in [self.OPEN_PALM, self.CLOSED_FIST]
        event = self.detect_gesture(current_state, hand_visible)
        
        return current_state, event, frame, palm_facing


class HandCopyPasteApp(ctk.CTk):
    """Modern UI Application"""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("Hand Copy Paste")
        self.geometry("1000x700")
        self.minsize(900, 600)
        
        # Initialize components
        self.gesture_detector = HandGestureDetector()
        self.firebase = FirebaseService()
        
        # State
        self.room_code = None
        self.is_connected = False
        self.cap = None
        self.running = False
        self.content_preview = None
        self.transfer_history = []
        
        # Load settings
        self.settings_file = "settings.json"
        settings = self._load_settings()
        self.save_folder = settings.get("save_folder", os.getcwd())
        self.last_room = settings.get("last_room", None)
        
        # Processing state
        self.is_processing = False
        
        # Periodic content check
        self.content_check_timer = None
        
        # Create UI
        self._create_ui()
        
        # Connect to Firebase and auto-rejoin last room
        self._connect_firebase()
        
        # Start camera
        self._start_camera()
        
        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_ui(self):
        """Create the modern UI layout"""
        
        # Main container
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Left panel - Camera feed
        self.left_panel = ctk.CTkFrame(self, corner_radius=10)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Camera header
        self.camera_header = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.camera_header.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        
        self.title_label = ctk.CTkLabel(
            self.camera_header, 
            text="✋ Hand Copy Paste",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(side="left")
        
        self.status_indicator = ctk.CTkLabel(
            self.camera_header,
            text="● Connecting...",
            font=ctk.CTkFont(size=12),
            text_color="orange"
        )
        self.status_indicator.pack(side="right", padx=10)
        
        # Camera canvas
        self.camera_frame = ctk.CTkFrame(self.left_panel, fg_color="#1a1a1a", corner_radius=10)
        self.camera_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Gesture status bar
        self.gesture_bar = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.gesture_bar.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
        
        self.hand_status = ctk.CTkLabel(
            self.gesture_bar,
            text="Hand: Not detected",
            font=ctk.CTkFont(size=14)
        )
        self.hand_status.pack(side="left", padx=10)
        
        self.holding_status = ctk.CTkLabel(
            self.gesture_bar,
            text="Status: Empty",
            font=ctk.CTkFont(size=14)
        )
        self.holding_status.pack(side="left", padx=20)
        
        self.ready_status = ctk.CTkLabel(
            self.gesture_bar,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#00d4aa"
        )
        self.ready_status.pack(side="right", padx=10)
        
        # Right panel - Controls
        self.right_panel = ctk.CTkFrame(self, corner_radius=10)
        self.right_panel.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        
        # Room section
        self.room_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.room_frame.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(
            self.room_frame,
            text="Room",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w")
        
        self.room_code_label = ctk.CTkLabel(
            self.room_frame,
            text="Not connected",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#666666"
        )
        self.room_code_label.pack(pady=10)
        
        # QR Code display
        self.qr_label = ctk.CTkLabel(self.room_frame, text="")
        self.qr_label.pack(pady=5)
        
        # Room buttons
        self.room_buttons = ctk.CTkFrame(self.room_frame, fg_color="transparent")
        self.room_buttons.pack(fill="x", pady=10)
        
        self.create_room_btn = ctk.CTkButton(
            self.room_buttons,
            text="Create Room",
            command=self._create_room,
            fg_color="#00aa88",
            hover_color="#008866",
            height=40
        )
        self.create_room_btn.pack(fill="x", pady=2)
        
        self.join_room_btn = ctk.CTkButton(
            self.room_buttons,
            text="Join Room",
            command=self._show_join_dialog,
            fg_color="#0088cc",
            hover_color="#006699",
            height=40
        )
        self.join_room_btn.pack(fill="x", pady=2)
        
        # Divider
        ctk.CTkFrame(self.right_panel, height=2, fg_color="#333333").pack(fill="x", padx=15, pady=10)
        
        # Content section
        self.content_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.content_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            self.content_frame,
            text="Content",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w")
        
        self.content_type_var = ctk.StringVar(value="screenshot")
        
        self.screenshot_radio = ctk.CTkRadioButton(
            self.content_frame,
            text="Screenshot",
            variable=self.content_type_var,
            value="screenshot"
        )
        self.screenshot_radio.pack(anchor="w", pady=2)
        
        self.clipboard_radio = ctk.CTkRadioButton(
            self.content_frame,
            text="Clipboard",
            variable=self.content_type_var,
            value="clipboard"
        )
        self.clipboard_radio.pack(anchor="w", pady=2)
        
        # Preview
        self.preview_frame = ctk.CTkFrame(self.content_frame, fg_color="#1a1a1a", height=100, corner_radius=8)
        self.preview_frame.pack(fill="x", pady=10)
        self.preview_frame.pack_propagate(False)
        
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="No content",
            text_color="#666666"
        )
        self.preview_label.pack(expand=True)
        
        # Divider
        ctk.CTkFrame(self.right_panel, height=2, fg_color="#333333").pack(fill="x", padx=15, pady=10)
        
        # Save folder section
        self.save_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.save_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            self.save_frame,
            text="Save Location",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w")
        
        self.save_folder_label = ctk.CTkLabel(
            self.save_frame,
            text=self._truncate_path(self.save_folder),
            font=ctk.CTkFont(size=10),
            text_color="#888888"
        )
        self.save_folder_label.pack(anchor="w", pady=(2, 5))
        
        self.change_folder_btn = ctk.CTkButton(
            self.save_frame,
            text="📁 Change Folder",
            command=self._change_save_folder,
            fg_color="#444444",
            hover_color="#555555",
            height=30
        )
        self.change_folder_btn.pack(fill="x")
        
        # Divider
        ctk.CTkFrame(self.right_panel, height=2, fg_color="#333333").pack(fill="x", padx=15, pady=10)
        
        # History section
        self.history_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.history_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        ctk.CTkLabel(
            self.history_frame,
            text="History",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w")
        
        self.history_list = ctk.CTkScrollableFrame(
            self.history_frame,
            fg_color="#1a1a1a",
            corner_radius=8
        )
        self.history_list.pack(fill="both", expand=True, pady=5)
        
        # Instructions
        self.instructions = ctk.CTkLabel(
            self.right_panel,
            text="✊ Close fist = GRAB  |  ✋ Open palm = DROP",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.instructions.pack(pady=10)
    
    def _connect_firebase(self):
        """Connect to Firebase in background"""
        def connect():
            if self.firebase.connect():
                self.is_connected = True
                self.after(0, lambda: self.status_indicator.configure(
                    text="● Connected",
                    text_color="#00d4aa"
                ))
                # Auto-rejoin last room
                if self.last_room:
                    self.after(100, lambda: self._auto_rejoin_room(self.last_room))
            else:
                self.after(0, lambda: self.status_indicator.configure(
                    text="● Offline",
                    text_color="#ff6666"
                ))
        
        threading.Thread(target=connect, daemon=True).start()
    
    def _auto_rejoin_room(self, room_code):
        """Automatically rejoin the last room"""
        if self.firebase.join_room(room_code):
            self.room_code = room_code
            self.room_code_label.configure(text=room_code, text_color="#00d4aa")
            self._generate_qr(room_code)
            self._add_history("Auto-joined", f"Room {room_code}")
            # Start periodic content checking
            self._start_periodic_content_check()
        else:
            # Room no longer exists, clear saved room
            self.last_room = None
            self._save_settings()
            self._add_history("Info", "Previous room expired")
    
    def _check_and_sync_content_state(self):
        """Check if content exists in room and sync gesture state"""
        def check_content():
            try:
                # Check if content exists in Firebase
                content = self.firebase.db.child("rooms").child(self.room_code).child("content").get()
                has_content = content.val() is not None
                
                # Update gesture state on main thread
                self.after(0, lambda: self._update_holding_state(has_content))
            except Exception as e:
                print(f"Content check error: {e}")
        
        threading.Thread(target=check_content, daemon=True).start()
    
    def _update_holding_state(self, has_content):
        """Update gesture holding state based on room content"""
        if has_content:
            self.gesture_detector.is_holding = True
            self.gesture_detector.confirmed_state = self.gesture_detector.CLOSED_FIST
            self._add_history("Info", "Content available - open palm to receive")
            print("📦 Content detected in room - ready to DROP")
        else:
            self.gesture_detector.is_holding = False
            self.gesture_detector.confirmed_state = self.gesture_detector.UNKNOWN
    
    def _start_periodic_content_check(self):
        """Start periodic content checking every 2 seconds"""
        if self.room_code:
            self._check_and_sync_content_state()
            # Schedule next check in 2 seconds
            self.content_check_timer = self.after(2000, self._start_periodic_content_check)
    
    def _stop_periodic_content_check(self):
        """Stop periodic content checking"""
        if self.content_check_timer:
            self.after_cancel(self.content_check_timer)
            self.content_check_timer = None
    
    def _start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self._update_camera()
    
    def _update_camera(self):
        """Update camera feed"""
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Process gestures
            state, event, frame, palm_facing = self.gesture_detector.process_frame(frame)
            
            # Handle events
            if event == self.gesture_detector.GRAB:
                self._on_grab()
            elif event == self.gesture_detector.DROP:
                self._on_drop()
            
            # Draw overlay
            frame = self._draw_overlay(frame, state, event, palm_facing)
            
            # Update UI labels
            self._update_status_labels(state)
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((580, 435), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo
        
        self.after(16, self._update_camera)  # ~60 FPS
    
    def _draw_overlay(self, frame, state, event, palm_facing):
        """Draw overlay on camera frame"""
        h, w = frame.shape[:2]
        
        # Event animation
        if event in [self.gesture_detector.GRAB, self.gesture_detector.DROP]:
            event_time = time.time() - self.gesture_detector.last_event_time
            if event_time < 0.8:
                alpha = 1.0 - event_time / 0.8
                
                if self.gesture_detector.last_event == self.gesture_detector.GRAB:
                    text, color = "GRAB!", (0, 200, 255)
                else:
                    text, color = "DROP!", (0, 255, 150)
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//4, h//2-50), (3*w//4, h//2+50), (0, 0, 0), -1)
                cv2.addWeighted(overlay, alpha*0.7, frame, 1-alpha*0.7, 0, frame)
                
                font_scale = 2.5 - event_time * 2
                adj_color = tuple(int(c * alpha) for c in color)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
                tx = (w - text_size[0]) // 2
                cv2.putText(frame, text, (tx, h//2+15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, adj_color, 3)
        
        # Holding indicator
        if self.gesture_detector.is_holding:
            cv2.rectangle(frame, (w-60, 10), (w-10, 60), (0, 200, 255), -1)
            cv2.putText(frame, "HELD", (w-55, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return frame
    
    def _update_status_labels(self, state):
        """Update status labels"""
        # Hand state
        if state == self.gesture_detector.OPEN_PALM:
            self.hand_status.configure(text="Hand: Open Palm", text_color="#00d4aa")
        elif state == self.gesture_detector.CLOSED_FIST:
            self.hand_status.configure(text="Hand: Closed Fist", text_color="#ffaa00")
        else:
            self.hand_status.configure(text="Hand: Not detected", text_color="#888888")
        
        # Holding state
        if self.gesture_detector.is_holding:
            self.holding_status.configure(text="Status: Holding", text_color="#ffaa00")
        else:
            self.holding_status.configure(text="Status: Empty", text_color="#888888")
        
        # Ready state
        if not self.gesture_detector.is_holding and self.gesture_detector.grab_ready:
            self.ready_status.configure(text="Ready to GRAB ✊")
        elif self.gesture_detector.is_holding and self.gesture_detector.drop_ready:
            self.ready_status.configure(text="Ready to DROP ✋")
        else:
            self.ready_status.configure(text="")
    
    def _create_room(self):
        """Create a new room"""
        if not self.is_connected:
            self._show_message("Not connected to server")
            return
        
        room_code = self.firebase.create_room()
        if room_code:
            self.room_code = room_code
            self.room_code_label.configure(text=room_code, text_color="#00d4aa")
            self._generate_qr(room_code)
            self._add_history("Room created", room_code)
            self._save_settings()  # Remember this room
            self._start_periodic_content_check()  # Start checking for content
        else:
            self._show_message("Failed to create room")
    
    def _show_join_dialog(self):
        """Show dialog to join room"""
        dialog = ctk.CTkInputDialog(
            text="Enter room code:",
            title="Join Room"
        )
        code = dialog.get_input()
        
        if code:
            code = code.strip().upper()
            if len(code) == 6:
                self._join_room(code)
            else:
                self._show_message("Invalid room code")
    
    def _join_room(self, code):
        """Join an existing room"""
        if not self.is_connected:
            self._show_message("Not connected to server")
            return
        
        if self.firebase.join_room(code):
            self.room_code = code
            self.room_code_label.configure(text=code, text_color="#00d4aa")
            self._generate_qr(code)
            self._add_history("Joined room", code)
            self._save_settings()  # Remember this room
            self._start_periodic_content_check()  # Start checking for content
        else:
            self._show_message("Failed to join room")
    
    def _generate_qr(self, code):
        """Generate QR code for room"""
        qr = qrcode.QRCode(version=1, box_size=4, border=2)
        qr.add_data(f"handcopypaste:{code}")
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="#00d4aa", back_color="#1a1a1a")
        qr_img = qr_img.resize((100, 100), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(qr_img)
        self.qr_label.configure(image=photo)
        self.qr_label.image = photo
    
    def _on_grab(self):
        """Handle GRAB gesture"""
        if not self.room_code:
            self._show_message("Create or join a room first!")
            return
        
        if self.is_processing:
            return  # Already processing
        
        content_type = self.content_type_var.get()
        
        if content_type == "screenshot":
            # Capture screenshot (quick operation)
            content = self._capture_screenshot()
            if content:
                # Upload in background thread
                self.is_processing = True
                self._show_processing("📤 Uploading screenshot...")
                
                def upload_task():
                    try:
                        success = self.firebase.upload_content(content, "image")
                        # Update UI on main thread
                        self.after(0, lambda: self._on_upload_complete(success, content, "Screenshot"))
                    except Exception as e:
                        self.after(0, lambda: self._on_upload_error(str(e)))
                
                threading.Thread(target=upload_task, daemon=True).start()
        else:
            # Clipboard (quick operation, no need for thread)
            try:
                text = pyperclip.paste()
                if text:
                    self.is_processing = True
                    self._show_processing("📤 Uploading text...")
                    
                    def upload_text_task():
                        try:
                            success = self.firebase.upload_content(text, "text")
                            self.after(0, lambda: self._on_text_upload_complete(success, text))
                        except Exception as e:
                            self.after(0, lambda: self._on_upload_error(str(e)))
                    
                    threading.Thread(target=upload_text_task, daemon=True).start()
            except:
                pass
    
    def _on_upload_complete(self, success, content, content_type):
        """Called when upload finishes"""
        self.is_processing = False
        self._hide_processing()
        
        if success:
            self._add_history("GRAB", f"{content_type} uploaded ✓")
            self._update_preview(content)
            # Set holding state after successful upload
            self.gesture_detector.is_holding = True
        else:
            self._add_history("GRAB", "Upload blocked - content already in room")
    
    def _on_text_upload_complete(self, success, text):
        """Called when text upload finishes"""
        self.is_processing = False
        self._hide_processing()
        
        if success:
            self._add_history("GRAB", f"Text: {text[:30]}... ✓")
            self.preview_label.configure(text=f"📋 {text[:50]}...")
        else:
            self._add_history("GRAB", "Upload blocked - content already in room")
    
    def _on_upload_error(self, error):
        """Called when upload has an error"""
        self.is_processing = False
        self._hide_processing()
        self._add_history("GRAB", f"Error: {error}")
    
    def _on_drop(self):
        """Handle DROP gesture"""
        if not self.room_code:
            self._show_message("Create or join a room first!")
            return
        
        if self.is_processing:
            return  # Already processing
        
        self.is_processing = True
        self._show_processing("📥 Downloading content...")
        
        def download_task():
            try:
                content_data = self.firebase.download_content()
                # Process on main thread
                self.after(0, lambda: self._on_download_complete(content_data))
            except Exception as e:
                self.after(0, lambda: self._on_download_error(str(e)))
        
        threading.Thread(target=download_task, daemon=True).start()
    
    def _on_download_complete(self, content_data):
        """Called when download finishes - works even after app restart"""
        self.is_processing = False
        self._hide_processing()
        
        if content_data:
            content_type = content_data.get("type", "unknown")
            
            if content_type == "image":
                try:
                    # Get image data (from Storage it's raw bytes, from DB it might be base64)
                    if content_data.get("raw_bytes"):
                        img_data = content_data["data"]  # Direct PNG bytes from Storage
                    else:
                        # Legacy: base64 encoded
                        img_data = base64.b64decode(content_data["data"])
                        # Check if zlib compressed
                        if content_data.get("compressed", False):
                            try:
                                img_data = zlib.decompress(img_data)
                            except:
                                pass  # Not compressed
                    
                    # Save to configured folder
                    filename = f"received_{int(time.time())}.png"
                    filepath = os.path.join(self.save_folder, filename)
                    
                    # Ensure folder exists
                    os.makedirs(self.save_folder, exist_ok=True)
                    
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    
                    # Get image dimensions
                    from PIL import Image as PILImage
                    img = PILImage.open(BytesIO(img_data))
                    
                    self._add_history("DROP", f"Saved: {filename} ({img.size[0]}x{img.size[1]}, {len(img_data)/1024:.1f}KB) ✓")
                    self._update_preview(img_data)
                    
                    # Open the image
                    self._open_file(filepath)
                    
                    # Reset holding state after successful download
                    self.gesture_detector.is_holding = False
                    
                except Exception as e:
                    self._add_history("DROP", f"Error: {e}")
            
            elif content_type == "text":
                text = content_data.get("data", "")
                pyperclip.copy(text)
                self._add_history("DROP", f"Copied: {text[:30]}... ✓")
                self.preview_label.configure(text=f"📋 {text[:50]}...")
                # Reset holding state after successful download
                self.gesture_detector.is_holding = False
        else:
            self._add_history("DROP", "No content available")
    
    def _on_download_error(self, error):
        """Called when download has an error"""
        self.is_processing = False
        self._hide_processing()
        self._add_history("DROP", f"Error: {error}")
    
    def _capture_screenshot(self):
        """Capture screenshot - Full resolution, LOSSLESS PNG for Storage"""
        try:
            # Completely hide window (not just minimize)
            self.withdraw()  # Hide from screen completely
            self.update()    # Process all pending events
            time.sleep(0.3)  # Wait for window to fully disappear
            self.update()    # Ensure redraw happened
            
            # Capture full resolution screenshot
            screenshot = ImageGrab.grab()
            
            # Restore window
            self.deiconify()
            self.lift()      # Bring to front
            self.focus_force()
            self.update()
            
            # Save as PNG (LOSSLESS) - Firebase Storage handles the transfer
            buffer = BytesIO()
            screenshot.save(buffer, format="PNG", optimize=True, compress_level=9)
            png_data = buffer.getvalue()
            
            print(f"Screenshot: {screenshot.size[0]}x{screenshot.size[1]}, PNG: {len(png_data)/1024:.1f}KB (lossless)")
            return png_data
        except Exception as e:
            print(f"Screenshot error: {e}")
            self.deiconify()
            return None
    
    def _update_preview(self, img_bytes):
        """Update content preview"""
        try:
            img = Image.open(BytesIO(img_bytes))
            img.thumbnail((150, 90), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo
        except:
            self.preview_label.configure(text="Preview unavailable")
    
    def _add_history(self, action, detail):
        """Add item to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        entry = ctk.CTkFrame(self.history_list, fg_color="#2a2a2a", corner_radius=5)
        entry.pack(fill="x", pady=2, padx=2)
        
        color = "#00d4aa" if action == "GRAB" else "#ffaa00" if action == "DROP" else "#888888"
        
        ctk.CTkLabel(
            entry,
            text=f"[{timestamp}] {action}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=color
        ).pack(anchor="w", padx=5, pady=(3, 0))
        
        ctk.CTkLabel(
            entry,
            text=detail,
            font=ctk.CTkFont(size=10),
            text_color="#888888"
        ).pack(anchor="w", padx=5, pady=(0, 3))
    
    def _show_message(self, message):
        """Show temporary message"""
        self._add_history("Info", message)
    
    def _show_processing(self, message):
        """Show processing indicator"""
        self.ready_status.configure(text=message, text_color="#ffaa00")
        self.update_idletasks()  # Force UI update
    
    def _hide_processing(self):
        """Hide processing indicator"""
        self.ready_status.configure(text="", text_color="#00d4aa")
    
    def _load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_settings(self):
        """Save settings to file"""
        try:
            settings = {
                "save_folder": self.save_folder,
                "last_room": self.room_code
            }
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def _change_save_folder(self):
        """Open folder picker to change save location"""
        folder = filedialog.askdirectory(
            title="Select Save Folder",
            initialdir=self.save_folder
        )
        if folder:
            self.save_folder = folder
            self.save_folder_label.configure(text=self._truncate_path(folder))
            self._save_settings()
            self._add_history("Settings", f"Save folder: {folder}")
    
    def _truncate_path(self, path, max_len=30):
        """Truncate path for display"""
        if len(path) <= max_len:
            return path
        return "..." + path[-(max_len-3):]
    
    def _open_file(self, filepath):
        """Open file with default system application"""
        try:
            # Windows
            os.startfile(filepath)
        except AttributeError:
            # macOS / Linux
            try:
                subprocess.run(["xdg-open", filepath], check=False)
            except:
                subprocess.run(["open", filepath], check=False)
        except Exception as e:
            print(f"Error opening file: {e}")
    
    def _on_close(self):
        """Handle window close"""
        self.running = False
        self._stop_periodic_content_check()
        if self.cap:
            self.cap.release()
        self.firebase.disconnect()
        self.destroy()


def main():
    app = HandCopyPasteApp()
    app.mainloop()


if __name__ == "__main__":
    main()
