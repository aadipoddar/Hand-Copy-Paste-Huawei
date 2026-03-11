"""
Firebase Service for Hand Copy Paste
=====================================
Handles Firebase Realtime Database + Firebase Storage for lossless images
"""

import pyrebase
import json
import os
import sys
import random
import string
import base64
import time
import uuid
import urllib.request


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


class FirebaseService:
    """Firebase service with Storage for lossless images"""
    
    def __init__(self):
        self.firebase = None
        self.db = None
        self.storage = None
        self.auth = None
        self.user = None
        self.user_token = None
        self.room_code = None
        self.device_id = self._get_device_id()
        self.connected = False
    
    def _get_device_id(self):
        """Get or create unique device ID"""
        id_file = ".device_id"
        if os.path.exists(id_file):
            with open(id_file, "r") as f:
                return f.read().strip()
        
        device_id = str(uuid.uuid4())[:8]
        with open(id_file, "w") as f:
            f.write(device_id)
        return device_id
    
    def connect(self):
        """Connect to Firebase"""
        try:
            config_path = get_resource_path("firebase_config.json")
            if not os.path.exists(config_path):
                print(f"firebase_config.json not found at {config_path}!")
                return False
            
            with open(config_path, "r") as f:
                config = json.load(f)
            
            self.firebase = pyrebase.initialize_app(config)
            self.db = self.firebase.database()
            self.storage = self.firebase.storage()
            self.auth = self.firebase.auth()
            
            # Anonymous sign in (needed for Storage)
            try:
                self.user = self.auth.sign_in_anonymous()
                self.user_token = self.user['idToken']
                print(f"Signed in anonymously: {self.user['localId'][:8]}...")
            except Exception as e:
                print(f"Auth warning (continuing): {e}")
                self.user_token = None
            
            self.connected = True
            print("Firebase connected successfully!")
            return True
            
        except Exception as e:
            print(f"Firebase connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Firebase"""
        self.connected = False
        self.room_code = None
    
    def create_room(self):
        """Create a new room"""
        if not self.connected:
            return None
        
        try:
            # Generate 6-character room code
            room_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            
            # Create room in database
            room_data = {
                "created_at": time.time(),
                "created_by": self.device_id,
                "members": {self.device_id: True},
                "content": None
            }
            
            self.db.child("rooms").child(room_code).set(room_data)
            self.room_code = room_code
            print(f"Room created: {room_code}")
            return room_code
            
        except Exception as e:
            print(f"Create room error: {e}")
            return None
    
    def join_room(self, room_code):
        """Join an existing room"""
        if not self.connected:
            return False
        
        try:
            room_code = room_code.upper().strip()
            
            # Check if room exists
            room = self.db.child("rooms").child(room_code).get()
            if not room.val():
                print(f"Room {room_code} not found")
                return False
            
            # Add device to room
            self.db.child("rooms").child(room_code).child("members").child(self.device_id).set(True)
            self.room_code = room_code
            print(f"Joined room: {room_code}")
            return True
            
        except Exception as e:
            print(f"Join room error: {e}")
            return False
    
    def upload_content(self, content, content_type):
        """Upload content - Storage for images, Database for text"""
        if not self.connected or not self.room_code:
            print("Not connected or no room")
            return False
        
        try:
            # CHECK: Do not upload if content already exists
            existing_content = self.db.child("rooms").child(self.room_code).child("content").get()
            if existing_content.val() is not None:
                print("❌ Content already present in room - cannot upload")
                return False
            
            timestamp = int(time.time() * 1000)
            
            if content_type == "image":
                # Upload PNG to Firebase Storage (lossless)
                storage_path = f"rooms/{self.room_code}/{timestamp}.png"
                
                # Upload to storage
                self.storage.child(storage_path).put(content, self.user_token)
                
                # Get download URL
                download_url = self.storage.child(storage_path).get_url(self.user_token)
                
                # Store metadata in database (not the image)
                content_data = {
                    "type": "image",
                    "storage_path": storage_path,
                    "download_url": download_url,
                    "uploaded_by": self.device_id,
                    "timestamp": timestamp,
                    "size_bytes": len(content),
                    "downloaded": False
                }
                
                print(f"Image uploaded to Storage: {len(content)/1024:.1f} KB")
                
            else:
                # Text content goes to database
                content_data = {
                    "type": "text",
                    "data": content,
                    "uploaded_by": self.device_id,
                    "timestamp": timestamp
                }
            
            # Save metadata to database
            self.db.child("rooms").child(self.room_code).child("content").set(content_data)
            print(f"Content uploaded: {content_type}")
            return True
            
        except Exception as e:
            print(f"Upload error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def download_content(self):
        """Download content from room - accessible by ANY device in the room"""
        if not self.connected or not self.room_code:
            return None
        
        try:
            content = self.db.child("rooms").child(self.room_code).child("content").get()
            content_data = content.val()
            
            if not content_data:
                print("No content in room")
                return None
            
            content_type = content_data.get("type")
            uploaded_by = content_data.get("uploaded_by", "unknown")
            
            # Log cross-device transfer
            if uploaded_by != self.device_id:
                print(f"📥 Receiving content from device: {uploaded_by}")
            else:
                print(f"📥 Receiving own content")
            
            if content_type == "image":
                download_url = content_data.get("download_url")
                storage_path = content_data.get("storage_path")
                
                if download_url:
                    # Download from Storage
                    print(f"Downloading from Storage...")
                    response = urllib.request.urlopen(download_url)
                    image_data = response.read()
                    
                    print(f"Image downloaded: {len(image_data)/1024:.1f} KB")
                    
                    # Delete from Storage after download (cleanup)
                    try:
                        self._delete_from_storage(storage_path)
                    except Exception as e:
                        print(f"Cleanup warning: {e}")
                    
                    # Clear content from database
                    self.db.child("rooms").child(self.room_code).child("content").remove()
                    
                    return {
                        "type": "image",
                        "data": image_data,
                        "raw_bytes": True,
                        "compressed": False  # Already PNG, not zlib compressed
                    }
                else:
                    print("No download URL found")
                    return None
                    
            elif content_type == "text":
                # Clear content after reading
                self.db.child("rooms").child(self.room_code).child("content").remove()
                return {
                    "type": "text",
                    "data": content_data.get("data", "")
                }
            
            return content_data
            
        except Exception as e:
            print(f"Download error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _delete_from_storage(self, storage_path):
        """Delete file from Firebase Storage using REST API"""
        try:
            # Use Firebase Storage REST API for deletion
            import urllib.parse
            
            # Load config to get bucket name
            with open("firebase_config.json", "r") as f:
                config = json.load(f)
            
            bucket = config.get("storageBucket")
            encoded_path = urllib.parse.quote(storage_path, safe='')
            
            delete_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o/{encoded_path}"
            
            req = urllib.request.Request(delete_url, method='DELETE')
            if self.user_token:
                req.add_header('Authorization', f'Bearer {self.user_token}')
            
            urllib.request.urlopen(req)
            print(f"✓ Deleted from Storage: {storage_path}")
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"File already deleted: {storage_path}")
            else:
                print(f"Storage delete error ({e.code}): {e.reason}")
        except Exception as e:
            print(f"Storage delete warning: {e}")
    
    def clear_room_content(self):
        """Clear content from room"""
        if not self.connected or not self.room_code:
            return False
        
        try:
            self.db.child("rooms").child(self.room_code).child("content").remove()
            return True
        except Exception as e:
            print(f"Clear error: {e}")
            return False
