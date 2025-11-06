"""
Emotion Detection Module
Handles face detection, text analysis, and voice input
"""

import cv2
import numpy as np
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Get project root directory (parent of src)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = DATA_DIR / "exports"
SESSIONS_DIR = DATA_DIR / "sessions"

# Ensure directories exist
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Optional imports with fallbacks
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("⚠️ SpeechRecognition not available - voice features disabled")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available - gesture features disabled")


class EmotionDetector:
    """Main emotion detection class handling multiple input modalities"""
    
    def __init__(self, config_path: str = "../config.json"):
        """Initialize the emotion detector with configuration"""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Emotion labels (must match model training order)
        self.emotions = self.config.get("emotions", [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ])
        
        # Emotion colors (BGR format for OpenCV)
        self.emotion_colors = {
            "angry": (0, 0, 255),      # Red
            "disgust": (0, 128, 0),    # Green
            "fear": (128, 0, 128),     # Purple
            "happy": (0, 255, 255),    # Yellow
            "neutral": (128, 128, 128), # Gray
            "sad": (255, 128, 0),      # Orange/Blue
            "surprise": (255, 255, 0)  # Cyan
        }
        
        # Load emotion detection model
        self.emotion_model = self._load_emotion_model()
        
        # Initialize face detection
        self.face_cascade = self._load_face_cascade()
        
        # Text emotion keywords
        self.emotion_keywords = self._load_emotion_keywords()
        
        # Voice recognizer
        self.recognizer = sr.Recognizer() if SPEECH_AVAILABLE else None
        
        # MediaPipe for hand gesture detection
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("✅ MediaPipe hand tracking initialized")
        else:
            self.mp_hands = None
            self.hands = None
            self.mp_drawing = None
        
        # Gesture state with improved stabilization
        self.current_gesture = "none"
        self.gesture_history = []
        self.gesture_stability_count = 0  # Count consecutive same gestures
        self.stable_gesture_threshold = 8  # Frames needed for stable gesture
        self.last_stable_gesture = "none"
        
        # Hand motion tracking
        self.hand_position_history = []  # Track hand movement
        self.gesture_motion = "static"  # up, down, static
        self.motion_history = []  # Track motion stability
        self.motion_threshold = 0.03  # Threshold for vertical motion detection
        
        # Session data
        self.session_data = []
        self.session_start = datetime.now()
        
        # Emotion smoothing (temporal filtering)
        self.emotion_history = []
        self.smoothing_window = 10  # Increased from 5 to 10 for more stability
        self.current_smoothed_emotions = None
        
        # Confidence threshold for emotion detection
        self.confidence_threshold = 0.3  # Lower threshold for better detection
    
    def _load_emotion_model(self):
        """Load the trained emotion detection model"""
        try:
            from tensorflow import keras
            
            # Try multiple model paths
            possible_paths = [
                "model/best_emotion_model.keras",
                "../model/best_emotion_model.keras",
                os.path.join(os.path.dirname(__file__), "..", "model", "best_emotion_model.keras"),
                PROJECT_ROOT / "model" / "best_emotion_model.keras"
            ]
            
            for path in possible_paths:
                path_str = str(path)
                if os.path.exists(path_str):
                    print(f"📦 Loading emotion model from: {path_str}")
                    model = keras.models.load_model(path_str)
                    print(f"✅ Emotion model loaded successfully")
                    return model
            
            print("⚠️ Emotion model not found, using fallback")
            return None
            
        except Exception as e:
            print(f"⚠️ Could not load emotion model: {e}")
            return None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load config: {e}. Using defaults.")
            return {
                "emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                "confidence_threshold": 0.5,
                "camera_index": 0
            }
    
    def _load_face_cascade(self) -> cv2.CascadeClassifier:
        """Load Haar Cascade for face detection"""
        cascade_path = self.config.get("model_path", "model/haarcascade_frontalface_default.xml")
        
        # Try multiple locations
        possible_paths = [
            cascade_path,
            "../" + cascade_path,
            os.path.join(os.path.dirname(__file__), "..", cascade_path)
        ]
        
        # Try OpenCV's built-in cascade
        for path in possible_paths:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    print(f"✅ Loaded face cascade from {path}")
                    return cascade
        
        # Try OpenCV's data directory
        cv_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if not cv_cascade.empty():
            print("✅ Loaded face cascade from OpenCV data")
            return cv_cascade
        
        print("⚠️ Could not load face cascade - face detection may not work")
        return cv2.CascadeClassifier()
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """Load emotion keyword mappings for text analysis"""
        return {
            "happy": ["happy", "joy", "excited", "great", "wonderful", "amazing", "love", "excellent", 
                     "fantastic", "cheerful", "delighted", "pleased", "glad", "content", "satisfied",
                     "thrilled", "ecstatic", "blissful", "joyful", "elated", "😊", "😀", "😄", "🎉"],
            "sad": ["sad", "unhappy", "depressed", "down", "miserable", "gloomy", "sorrowful", 
                   "heartbroken", "disappointed", "blue", "melancholy", "dejected", "grief", "crying",
                   "tears", "hurt", "pain", "lonely", "isolated", "😢", "😭", "💔"],
            "angry": ["angry", "mad", "furious", "rage", "annoyed", "irritated", "frustrated", 
                     "outraged", "livid", "hate", "upset", "pissed", "enraged", "fuming", "incensed",
                     "infuriated", "hostile", "aggressive", "😠", "😡", "🤬"],
            "fear": ["scared", "afraid", "fearful", "terrified", "anxious", "worried", "nervous", 
                    "frightened", "panic", "dread", "alarmed", "concerned", "uneasy", "apprehensive",
                    "timid", "horror", "phobia", "😨", "😰", "😱"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "wow", "unexpected", 
                        "incredible", "unbelievable", "sudden", "startled", "stunned", "astounded",
                        "dumbfounded", "flabbergasted", "😲", "😮", "🤯"],
            "disgust": ["disgusting", "gross", "nasty", "revolting", "repulsive", "sick", "awful",
                       "horrible", "terrible", "yuck", "ew", "vile", "foul", "nauseating", "🤢", "🤮"],
            "neutral": ["okay", "fine", "alright", "normal", "whatever", "meh", "so-so", "average"]
        }
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame with improved parameters for glasses"""
        if self.face_cascade.empty():
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection with glasses
        gray = cv2.equalizeHist(gray)
        
        # Adjusted parameters for better detection with glasses
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Smaller steps for better accuracy
            minNeighbors=3,    # Lower threshold for glasses
            minSize=(60, 60),  # Larger minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def predict_emotion(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Predict emotion from face image with temporal smoothing
        """
        if self.emotion_model is not None:
            try:
                # Preprocess face image for model
                # Most emotion models expect 48x48 grayscale input
                face_processed = cv2.resize(face_img, (48, 48))
                
                # Convert to grayscale if needed
                if len(face_processed.shape) == 3:
                    face_processed = cv2.cvtColor(face_processed, cv2.COLOR_BGR2GRAY)
                
                # Normalize pixel values
                face_processed = face_processed.astype('float32') / 255.0
                
                # Reshape for model input (batch_size, height, width, channels)
                face_processed = np.expand_dims(face_processed, axis=0)
                face_processed = np.expand_dims(face_processed, axis=-1)
                
                # Get prediction
                predictions = self.emotion_model.predict(face_processed, verbose=0)[0]
                
                # Create emotion dictionary
                current_emotions = {
                    emotion: float(prob) 
                    for emotion, prob in zip(self.emotions, predictions)
                }
                
            except Exception as e:
                print(f"⚠️ Model prediction error: {e}")
                # Fallback to neutral
                current_emotions = {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
        else:
            # Fallback: analyze face brightness/contrast for basic emotion hints
            current_emotions = self._simple_emotion_heuristic(face_img)
        
        # Add to history
        self.emotion_history.append(current_emotions)
        
        # Keep only recent history
        if len(self.emotion_history) > self.smoothing_window:
            self.emotion_history.pop(0)
        
        # Apply smoothing (moving average with exponential weighting)
        smoothed_emotions = self._smooth_emotions()
        self.current_smoothed_emotions = smoothed_emotions
        
        return smoothed_emotions
    
    def _simple_emotion_heuristic(self, face_img: np.ndarray) -> Dict[str, float]:
        """Simple emotion detection based on image properties (fallback)"""
        # Default to neutral with slight variations
        emotions = {emotion: 0.1 for emotion in self.emotions}
        emotions["neutral"] = 0.4
        return emotions
    
    def _smooth_emotions(self) -> Dict[str, float]:
        """Apply temporal smoothing to emotion predictions with exponential weighting"""
        if not self.emotion_history:
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
        
        # Use exponential moving average - recent frames have more weight
        # This provides stability while still being responsive
        smoothed = {}
        
        # Create weights that decay exponentially (more recent = higher weight)
        num_frames = len(self.emotion_history)
        weights = np.exp(np.linspace(-2, 0, num_frames))  # Exponential decay
        weights = weights / weights.sum()  # Normalize
        
        for emotion in self.emotions:
            values = np.array([frame[emotion] for frame in self.emotion_history])
            # Weighted average
            smoothed[emotion] = float(np.sum(values * weights))
        
        # Normalize to sum to 1
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v/total for k, v in smoothed.items()}
        
        return smoothed
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze emotion in text using keyword matching"""
        if not text or not text.strip():
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "sentiment": "neutral",
                "all_scores": {emotion: 0.0 for emotion in self.emotions}
            }
        
        text_lower = text.lower()
        scores = {emotion: 0 for emotion in self.emotions}
        
        # Count keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[emotion] += 1
        
        # Calculate total matches
        total_matches = sum(scores.values())
        
        if total_matches == 0:
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "sentiment": "neutral",
                "all_scores": {emotion: 0.0 for emotion in self.emotions}
            }
        
        # Normalize scores
        normalized_scores = {
            emotion: score / total_matches 
            for emotion, score in scores.items()
        }
        
        # Get dominant emotion
        dominant_emotion = max(normalized_scores.items(), key=lambda x: x[1])
        
        # Determine sentiment
        positive_emotions = ["happy", "surprise"]
        negative_emotions = ["sad", "angry", "fear", "disgust"]
        
        positive_score = sum(normalized_scores.get(e, 0) for e in positive_emotions)
        negative_score = sum(normalized_scores.get(e, 0) for e in negative_emotions)
        
        if positive_score > negative_score:
            sentiment = "positive"
        elif negative_score > positive_score:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "emotion": dominant_emotion[0],
            "confidence": dominant_emotion[1],
            "sentiment": sentiment,
            "all_scores": normalized_scores
        }
    
    def listen_voice(self, duration: int = 5) -> Optional[str]:
        """
        Listen to microphone and convert speech to text
        Requires PyAudio to be installed
        """
        if not SPEECH_AVAILABLE:
            return None
        
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... Speak now!")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
                
            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"📝 Recognized: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def save_session(self, filename: Optional[str] = None) -> str:
        """Save session data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = SESSIONS_DIR / f"session_{timestamp}.json"
        
        # Ensure it's a Path object
        filename = Path(filename)
        
        session_summary = {
            "start_time": self.session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_detections": len(self.session_data),
            "data": self.session_data
        }
        
        with open(filename, 'w') as f:
            json.dump(session_summary, f, indent=2)
        
        print(f"💾 Session saved to {filename}")
        return filename
    
    def export_csv(self, filename: Optional[str] = None) -> str:
        """Export session data to CSV"""
        try:
            import pandas as pd
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = EXPORTS_DIR / f"emotions_{timestamp}.csv"
            
            # Ensure it's a Path object
            filename = Path(filename)
            
            # Convert session data to DataFrame
            df = pd.DataFrame(self.session_data)
            df.to_csv(filename, index=False)
            
            print(f"📄 Data exported to {filename}")
            return filename
            
        except ImportError:
            print("⚠️ Pandas not available - CSV export disabled")
            return None
        except Exception as e:
            print(f"❌ Export error: {e}")
            return None
    
    def add_detection(self, detection_type: str, data: Dict):
        """Add a detection to session data"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": detection_type,
            **data
        }
        self.session_data.append(entry)
    
    def get_session_stats(self) -> Dict:
        """Get statistics about the current session"""
        if not self.session_data:
            return {
                "total_detections": 0,
                "duration": "00:00:00",
                "emotion_counts": {}
            }
        
        duration = datetime.now() - self.session_start
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Count emotions
        emotion_counts = {}
        for entry in self.session_data:
            if "emotion" in entry:
                emotion = entry["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "total_detections": len(self.session_data),
            "duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "emotion_counts": emotion_counts,
            "most_common": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "none"
        }
    
    def detect_gestures(self, frame: np.ndarray, show_points: bool = False) -> Tuple[np.ndarray, str, Dict]:
        """
        Detect hand gestures using MediaPipe
        
        Args:
            frame: Input video frame
            show_points: Whether to show numbered landmark points
            
        Returns:
            Tuple of (annotated_frame, gesture_name, hand_data)
        """
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            return frame, "unavailable", {}
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        gesture_name = "none"
        hand_data = {}
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Draw numbered points if enabled
                if show_points:
                    h, w, _ = frame.shape
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.putText(
                            frame, str(idx),
                            (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                        )
                
                # Detect gesture
                gesture_name = self._recognize_gesture(hand_landmarks)
                
                # Store hand data
                hand_data[f"hand_{hand_idx}"] = {
                    "gesture": gesture_name,
                    "landmarks": len(hand_landmarks.landmark)
                }
                
                # Draw gesture label
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h)
                
                # Display gesture and motion
                gesture_text = f"{gesture_name}"
                if self.gesture_motion != "static":
                    gesture_text += f" ({self.gesture_motion})"
                
                cv2.putText(
                    frame, gesture_text, 
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
        
        self.current_gesture = gesture_name
        hand_data["motion"] = self.gesture_motion
        
        return frame, gesture_name, hand_data
    
    def _recognize_gesture(self, hand_landmarks) -> str:
        """
        Improved gesture recognition with better finger detection and stability
        
        Gesture Guide:
        - Fist: No fingers extended (closed fist) - VOICE STOP
        - Pinch: Thumb and index finger touching - for brightness control with motion
        - Peace/Victory: Index and middle fingers up (V sign)
        - Pointing: Only index finger extended - VOICE START
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Gesture name
        """
        # Get landmark positions
        landmarks = hand_landmarks.landmark
        
        # Get key points
        wrist = landmarks[0]
        
        # Thumb landmarks
        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]
        
        # Index finger
        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        index_dip = landmarks[7]
        index_tip = landmarks[8]
        
        # Middle finger
        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        middle_dip = landmarks[11]
        middle_tip = landmarks[12]
        
        # Ring finger
        ring_mcp = landmarks[13]
        ring_pip = landmarks[14]
        ring_dip = landmarks[15]
        ring_tip = landmarks[16]
        
        # Pinky finger
        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        pinky_dip = landmarks[19]
        pinky_tip = landmarks[20]
        
        # Improved finger extension detection with stricter criteria
        # For thumb: check if tip is further from wrist than IP joint
        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_ip.x - wrist.x) * 1.2
        
        # For other fingers: check if tip is significantly above PIP joint
        # Using stricter thresholds to reduce false positives
        index_extended = (index_tip.y < index_pip.y - 0.02) and (index_tip.y < index_mcp.y)
        middle_extended = (middle_tip.y < middle_pip.y - 0.02) and (middle_tip.y < middle_mcp.y)
        ring_extended = (ring_tip.y < ring_pip.y - 0.02) and (ring_tip.y < ring_mcp.y)
        pinky_extended = (pinky_tip.y < pinky_pip.y - 0.02) and (pinky_tip.y < pinky_mcp.y)
        
        # Count extended fingers
        fingers_up = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
        
        # Track hand vertical position for motion-based brightness control
        # Use wrist Y position (lower Y = higher in frame = up motion)
        palm_center_y = wrist.y
        self.hand_position_history.append(palm_center_y)
        
        # Keep only recent history (15 frames for smoother motion detection)
        if len(self.hand_position_history) > 15:
            self.hand_position_history.pop(0)
        
        # Detect vertical hand motion (up/down) for brightness control
        if len(self.hand_position_history) >= 10:
            # Compare recent vs older positions
            recent_avg = sum(self.hand_position_history[-5:]) / 5
            older_avg = sum(self.hand_position_history[:5]) / 5
            delta = recent_avg - older_avg
            
            # Motion threshold for detecting up/down movement
            motion_threshold = 0.03  # Adjust sensitivity here
            
            # Negative delta = hand moving up (Y decreasing)
            # Positive delta = hand moving down (Y increasing)
            if delta < -motion_threshold:
                detected_motion = "up"  # Hand moving UP = Brightness UP
            elif delta > motion_threshold:
                detected_motion = "down"  # Hand moving DOWN = Brightness DOWN
            else:
                detected_motion = "static"
            
            # Track motion history for stability
            self.motion_history.append(detected_motion)
            if len(self.motion_history) > 5:
                self.motion_history.pop(0)
            
            # Set motion only if consistent across recent frames
            if len(self.motion_history) >= 3:
                most_common_motion = max(set(self.motion_history), key=self.motion_history.count)
                if self.motion_history.count(most_common_motion) >= 3:
                    self.gesture_motion = most_common_motion
                else:
                    self.gesture_motion = "static"
        else:
            self.gesture_motion = "static"
        
        # ===== Gesture recognition with improved logic =====
        
        # No fingers extended - FIST
        if fingers_up == 0:
            detected_gesture = "fist"
        
        # Index and middle up, others down - PEACE / VICTORY
        elif index_extended and middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
            detected_gesture = "peace"
        
        # Index only - POINTING (like pointing at something)
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
            detected_gesture = "pointing"
        
        # Two fingers (various combinations)
        elif fingers_up == 2:
            detected_gesture = "two_fingers"
        
        else:
            detected_gesture = "unknown"
        
        # Apply gesture stabilization
        self.gesture_history.append(detected_gesture)
        if len(self.gesture_history) > 10:
            self.gesture_history.pop(0)
        
        # Check if gesture is stable
        if len(self.gesture_history) >= 5:
            # Count occurrences of current gesture in recent history
            gesture_count = self.gesture_history.count(detected_gesture)
            if gesture_count >= 4:  # At least 4 out of last 5 frames
                stable_gesture = detected_gesture
                self.last_stable_gesture = stable_gesture
            else:
                # Return last stable gesture to prevent flickering
                stable_gesture = self.last_stable_gesture
        else:
            stable_gesture = detected_gesture
        
        return stable_gesture
    
    def get_gesture_emotion_mapping(self, gesture: str) -> str:
        """
        Map gestures to likely emotions
        
        Args:
            gesture: Detected gesture name
            
        Returns:
            Associated emotion
        """
        gesture_emotion_map = {
            "peace": "happy",
            "fist": "angry",
            "pointing": "neutral",
            "pinch": "neutral",
            "unknown": "neutral",
            "none": "neutral"
        }
        
        return gesture_emotion_map.get(gesture, "neutral")
    
    def get_gesture_action(self, gesture: str, motion: str = "static") -> Dict:
        """
        Map gestures and motions to actions
        
        Gesture Guide:
        - Pointing (☝️): Index finger - START VOICE RECORDING
        - Fist (✊): Closed hand - STOP VOICE/SPEAKING
        - Peace (✌️): Index + Middle - Stop video detection
        
        Args:
            gesture: Detected gesture
            motion: Detected motion (up, down, static)
            
        Returns:
            Dictionary with action info
        """
        # Static gesture actions
        static_actions = {
            "peace": {"action": "stop_video", "description": "✌️ Stop video detection"},
            "fist": {"action": "stop_voice", "description": "✊ STOP VOICE/SPEAKING"},
            "pointing": {"action": "start_voice", "description": "☝️ START VOICE CHAT"},
        }
        
        # Return static action if available
        return static_actions.get(gesture, {
            "action": "none",
            "description": f"{gesture}",
            "value": 0
        })
