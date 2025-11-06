"""
Emotion Detection System - Main GUI Application
Clean, simple, and performant interface using CustomTkinter
"""

import customtkinter as ctk
import cv2
from PIL import Image
import threading
import json
import os
import time
from datetime import datetime
from typing import Optional
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import requests
import io
import numpy as np

from detector import EmotionDetector
from gemini_chatbot import GeminiChatbot

# Optional speech recognition (used for manual mic control)
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️ SpeechRecognition not available - advanced voice capture disabled")

# Try to import pygame for audio playback
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ pygame not available - audio playback disabled")

# Try to import pyttsx3 as fallback TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️ pyttsx3 not available - fallback TTS disabled")

# Get project root directory (parent of src)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = DATA_DIR / "exports"

# Ensure directories exist
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class EmotionDetectionApp(ctk.CTk):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("🎯 Emotion Detection System")
        self.geometry("1400x800")
        
        # Initialize components
        self.detector = EmotionDetector()
        self.chatbot = GeminiChatbot()
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.current_emotion = "neutral"
        self.current_gesture = "none"
        self.current_motion = "static"
        
        # Input source mode
        self.input_mode = "webcam"  # webcam or image
        self.uploaded_image = None
        
        # Brightness control
        self.brightness = 0  # -100 to +100
        self.last_gesture_action_time = 0
        self.gesture_action_cooldown = 2.0  # seconds between actions
        
        # Gesture stability tracking
        self.gesture_history = []
        self.gesture_stability_frames = 5  # Require same gesture for 5 frames
        
        # AI Vision auto-analysis (enabled by default)
        self.auto_analysis_enabled = True
        self.last_analysis_time = 0
        self.analysis_interval = 30  # seconds between auto-analysis
        
        # Voice-to-text chat
        self.is_voice_recording = False
        self.voice_thread = None
        self.voice_stop_event = None
        self.voice_processing = False
        self.voice_max_duration = 10  # seconds
        
        # ElevenLabs TTS
        self.elevenlabs_api_key = self.detector.config.get("elevenlabs_api_key", "")
        self.is_speaking = False
        self.tts_enabled = False  # Toggle for TTS - default OFF
        self.use_elevenlabs = False  # Will be set to True if API key works
        self.elevenlabs_status = "unknown"  # unknown, working, no_permission, quota_exceeded, error
        
        # Initialize pyttsx3 as fallback
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)  # Speed
                self.tts_engine.setProperty('volume', 0.9)  # Volume
            except:
                self.tts_engine = None
        else:
            self.tts_engine = None

        # --- Voice options ---
        # Preset ElevenLabs voices (friendly name -> voice_id)
        # Common ElevenLabs premade voices
        self.available_eleven_voices = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Domi": "AZnzlk1XvdvUeBnXmlld",
            "Bella": "EXAVITQu4vr4xnSDxMaL",
            "Antoni": "ErXwobaYiN019PkySvjV",
            "Elli": "MF3mGyEYCl7XYWbV9V6O",
            "Josh": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold": "VR6AewLTigWG4xSOukaG",
            "Adam": "pNInz6obpgDQGcFmaJgB",
            "Sam": "yoZ06aMxZJJ28mfd3POQ"
        }

        # Default selections (can be overridden by config)
        self.eleven_voice_name = self.detector.config.get("eleven_voice_name", "Rachel")
        self.eleven_voice_id_custom = self.detector.config.get("eleven_voice_id_custom", "")

        # Build pyttsx3 voice map (name -> id) if engine available
        self.pyttsx3_voice_map = {}
        if self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices') or []
                for v in voices:
                    # use voice.name where available as human-friendly key
                    name = getattr(v, 'name', None) or getattr(v, 'id', str(v))
                    self.pyttsx3_voice_map[name] = getattr(v, 'id', str(v))

                # Pick first available as default
                first_voice_name = next(iter(self.pyttsx3_voice_map), None)
                self.pyttsx3_selected_voice = self.detector.config.get("pyttsx3_voice", first_voice_name)
                # Apply selected offline voice to engine immediately
                try:
                    sel_name = self.pyttsx3_selected_voice
                    if sel_name and sel_name in self.pyttsx3_voice_map:
                        self.tts_engine.setProperty('voice', self.pyttsx3_voice_map[sel_name])
                except Exception:
                    pass
            except Exception:
                self.pyttsx3_voice_map = {}
                self.pyttsx3_selected_voice = None
        else:
            self.pyttsx3_selected_voice = None
        
        # UI elements
        self.video_label = None
        self.emotion_bars = {}
        self.chat_display = None
        
        # Create UI
        self.create_widgets()
        
        # Load emotion colors
        self.emotion_colors = self.detector.config.get("emotion_colors", {})
        
    def create_widgets(self):
        """Create all UI widgets"""
        
        # Main container with 3 columns
        self.grid_columnconfigure(0, weight=2)  # Video panel
        self.grid_columnconfigure(1, weight=1)  # Analysis panel
        self.grid_columnconfigure(2, weight=1)  # Chat panel
        self.grid_rowconfigure(1, weight=1)
        
        # ===== TOP CONTROL BAR =====
        control_frame = ctk.CTkFrame(self, height=60, fg_color="transparent")
        control_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 5))
        
        # Start/Stop buttons
        self.start_btn = ctk.CTkButton(
            control_frame, text="▶️ Start", command=self.start_detection,
            width=100, height=35, font=("Arial", 14, "bold")
        )
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(
            control_frame, text="⏹️ Stop", command=self.stop_detection,
            width=100, height=35, font=("Arial", 14, "bold"), state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)
        
        # Snapshot button
        self.snapshot_btn = ctk.CTkButton(
            control_frame, text="📸 Snapshot", command=self.take_snapshot,
            width=120, height=35
        )
        self.snapshot_btn.pack(side="left", padx=5)
        
        # Upload button
        self.upload_btn = ctk.CTkButton(
            control_frame, text="🖼️ Upload Image", command=self.upload_image,
            width=130, height=35
        )
        self.upload_btn.pack(side="left", padx=5)
        
        # AI Vision button
        self.ai_vision_btn = ctk.CTkButton(
            control_frame, text="🤖 AI Vision", command=self.analyze_current_frame,
            width=120, height=35, fg_color="#9333ea", hover_color="#7e22ce"
        )
        self.ai_vision_btn.pack(side="left", padx=5)
        
        # Gesture detection toggle (enabled by default)
        self.gesture_var = ctk.BooleanVar(value=True)
        self.gesture_toggle = ctk.CTkSwitch(
            control_frame, text="👋 Gestures", 
            variable=self.gesture_var,
            command=self.toggle_gesture_visibility,
            font=("Arial", 11)
        )
        self.gesture_toggle.pack(side="left", padx=5)
        
        # Hidden gesture points button (only visible when gestures enabled)
        self.gesture_points_btn = ctk.CTkButton(
            control_frame, text="📍 Points", command=self.toggle_gesture_points,
            width=80, height=35, fg_color="#7c3aed", hover_color="#6d28d9"
        )
        # Initially hidden
        self.show_gesture_points = False
        
        # Test Chatbot button
        self.test_chatbot_btn = ctk.CTkButton(
            control_frame, text="🤖 Test Chat", command=self.test_chatbot,
            width=120, height=35, fg_color="#10b981", hover_color="#059669"
        )
        self.test_chatbot_btn.pack(side="left", padx=5)
        
        # Test All Voices button
        self.test_voices_btn = ctk.CTkButton(
            control_frame, text="🎤 Test Voice", command=self.test_all_voices,
            width=120, height=35, fg_color="#7c3aed", hover_color="#6d28d9"
        )
        self.test_voices_btn.pack(side="left", padx=5)
        
        # Help button
        self.help_btn = ctk.CTkButton(
            control_frame, text="❓ Help", command=self.show_help,
            width=80, height=35, fg_color="#0ea5e9", hover_color="#0284c7"
        )
        self.help_btn.pack(side="left", padx=5)
        
        # Settings button
        self.settings_btn = ctk.CTkButton(
            control_frame, text="⚙️ Settings", command=self.show_settings,
            width=100, height=35, fg_color="#8b5cf6", hover_color="#7c3aed"
        )
        self.settings_btn.pack(side="left", padx=5)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            control_frame, text="Ready", font=("Arial", 12)
        )
        self.status_label.pack(side="left", padx=20)
        
        # Export buttons (right side)
        self.export_csv_btn = ctk.CTkButton(
            control_frame, text="📄 Export CSV", command=self.export_csv,
            width=110, height=35
        )
        self.export_csv_btn.pack(side="right", padx=5)
        
        self.save_btn = ctk.CTkButton(
            control_frame, text="💾 Save", command=self.save_session,
            width=100, height=35
        )
        self.save_btn.pack(side="right", padx=5)
        
        # ===== LEFT PANEL: VIDEO FEED =====
        video_frame = ctk.CTkFrame(self)
        video_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        
        # Video title
        ctk.CTkLabel(
            video_frame, text="📹 Live Feed", 
            font=("Arial", 16, "bold")
        ).pack(pady=5)
        
        # Video display
        self.video_label = ctk.CTkLabel(video_frame, text="Camera Off")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Gesture display
        self.gesture_label = ctk.CTkLabel(
            video_frame, text="👋 Gesture: None", 
            font=("Arial", 14, "bold"),
            fg_color="#1f1f1f", corner_radius=8
        )
        self.gesture_label.pack(pady=5, padx=10, fill="x")
        
        # Brightness control display
        brightness_frame = ctk.CTkFrame(video_frame, fg_color="transparent")
        brightness_frame.pack(pady=5, padx=10, fill="x")
        
        ctk.CTkLabel(
            brightness_frame, text="🔆 Brightness:",
            font=("Arial", 11)
        ).pack(side="left", padx=5)
        
        self.brightness_slider = ctk.CTkSlider(
            brightness_frame, from_=-100, to=100,
            command=self.set_brightness_manual, width=200
        )
        self.brightness_slider.set(0)
        self.brightness_slider.pack(side="left", padx=5)
        
        self.brightness_value_label = ctk.CTkLabel(
            brightness_frame, text="0%",
            font=("Arial", 11), width=50
        )
        self.brightness_value_label.pack(side="left")
        
        # ===== MIDDLE PANEL: ANALYSIS =====
        analysis_frame = ctk.CTkFrame(self)
        analysis_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Analysis title
        ctk.CTkLabel(
            analysis_frame, text="📊 Analysis", 
            font=("Arial", 16, "bold")
        ).pack(pady=5)
        
        # Current emotion display
        self.current_emotion_label = ctk.CTkLabel(
            analysis_frame, text="😐 neutral", 
            font=("Arial", 24, "bold")
        )
        self.current_emotion_label.pack(pady=10)
        
        # Emotion bars container
        bars_container = ctk.CTkScrollableFrame(analysis_frame)
        bars_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Emotion color map for UI (RGB hex for CustomTkinter)
        emotion_ui_colors = {
            "angry": "#FF0000",      # Red
            "disgust": "#008000",    # Green
            "fear": "#800080",       # Purple
            "happy": "#FFFF00",      # Yellow
            "neutral": "#808080",    # Gray
            "sad": "#FF8000",        # Orange
            "surprise": "#00FFFF"    # Cyan
        }
        
        # Create emotion progress bars with colors
        for emotion in self.detector.emotions:
            emotion_frame = ctk.CTkFrame(bars_container, fg_color="transparent")
            emotion_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(
                emotion_frame, text=emotion.capitalize(),
                font=("Arial", 11), width=80, anchor="w"
            ).pack(side="left")
            
            # Create colored progress bar
            bar_color = emotion_ui_colors.get(emotion, "#00FF00")
            bar = ctk.CTkProgressBar(
                emotion_frame, width=200, height=15,
                progress_color=bar_color
            )
            bar.pack(side="left", padx=5)
            bar.set(0)
            self.emotion_bars[emotion] = bar
        
        # Smoothing control
        smoothing_frame = ctk.CTkFrame(bars_container, fg_color="transparent")
        smoothing_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            smoothing_frame, text="Smoothing:",
            font=("Arial", 11, "bold")
        ).pack(side="left", padx=5)
        
        self.smoothing_slider = ctk.CTkSlider(
            smoothing_frame, from_=1, to=20, number_of_steps=19,
            command=self.update_smoothing, width=150
        )
        self.smoothing_slider.set(10)  # Default increased to 10
        self.smoothing_slider.pack(side="left", padx=5)
        
        self.smoothing_label = ctk.CTkLabel(
            smoothing_frame, text="10 frames",
            font=("Arial", 10), width=70
        )
        self.smoothing_label.pack(side="left")
        
        # ===== RIGHT PANEL: AI CHAT =====
        chat_frame = ctk.CTkFrame(self)
        chat_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 10), pady=5)
        
        # Chat title
        ctk.CTkLabel(
            chat_frame, text="💬 Gemini AI Chat",
            font=("Arial", 16, "bold")
        ).pack(pady=5)
        
        # Personality selector
        personality_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        personality_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            personality_frame, text="Personality:",
            font=("Arial", 11)
        ).pack(side="left", padx=5)
        
        self.personality_var = ctk.StringVar(value="empathetic")
        personality_menu = ctk.CTkOptionMenu(
            personality_frame,
            values=["empathetic", "neutral", "professional"],
            variable=self.personality_var,
            command=self.change_personality,
            width=130
        )
        personality_menu.pack(side="left")
        
        # Voice selector frame
        tts_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        tts_frame.pack(fill="x", padx=10, pady=5)
        
        # Voice is always enabled by default
        self.tts_var = ctk.BooleanVar(value=True)
        self.tts_enabled = True
        
        # --- Unified Voice Selector ---
        voice_frame = ctk.CTkFrame(tts_frame, fg_color="transparent")
        voice_frame.pack(side="left", padx=(10, 0))

        ctk.CTkLabel(voice_frame, text="Voice:", font=("Arial", 11)).pack(side="left", padx=(0, 5))
        
        # Build combined voice list: ElevenLabs voices (if key exists) + Offline voices
        all_voices = []
        if self.elevenlabs_api_key:
            # Add ElevenLabs voices with [EL] prefix
            all_voices.extend([f"[EL] {name}" for name in self.available_eleven_voices.keys()])
            all_voices.append("[EL] Custom")
        
        # Add offline voices with [Offline] prefix
        if self.pyttsx3_voice_map:
            all_voices.extend([f"[Offline] {name}" for name in self.pyttsx3_voice_map.keys()])
        
        # Default to first available voice
        default_voice = all_voices[0] if all_voices else "No voices available"
        self.unified_voice_var = ctk.StringVar(value=default_voice)
        
        self.voice_menu = ctk.CTkOptionMenu(
            voice_frame,
            values=all_voices if all_voices else ["No voices available"],
            variable=self.unified_voice_var,
            command=lambda v: self.change_voice(v),
            width=200
        )
        self.voice_menu.pack(side="left")

        # Custom voice id entry (only shown when ElevenLabs Custom is selected)
        self.eleven_custom_entry = ctk.CTkEntry(voice_frame, placeholder_text="Paste custom voice id", width=220)
        # Initially hidden
        
        # Chat display
        self.chat_display = ctk.CTkTextbox(
            chat_frame, font=("Arial", 11), state="disabled"
        )
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Chat input
        chat_input_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        chat_input_frame.pack(fill="x", padx=10, pady=5)
        
        if self.detector.recognizer:
            # Ghost-style icon button for voice capture
            self.voice_btn = ctk.CTkButton(
                chat_input_frame,
                text="🎤",
                width=40,
                height=36,
                command=self.voice_input,
                fg_color="transparent",
                hover_color=("gray75", "gray25"),
                corner_radius=8
            )
            self.voice_btn.pack(side="left", padx=(0, 6))
        
        self.chat_input = ctk.CTkEntry(
            chat_input_frame, placeholder_text="Type your message...",
            font=("Arial", 11)
        )
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", lambda e: self.send_message())
        
        ctk.CTkButton(
            chat_input_frame, text="Send", 
            command=self.send_message, width=60
        ).pack(side="left")
        
        # Clear chat button
        ctk.CTkButton(
            chat_frame, text="Clear Chat", 
            command=self.clear_chat, height=30
        ).pack(pady=5)
    
    def start_detection(self):
        """Start video capture and emotion detection"""
        if self.is_running:
            return
        
        # Start webcam
        if not self._start_webcam():
            return
        
        # Common setup for webcam mode
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="🟢 Detecting...")
        
        # Show helpful tip
        self.add_chat_message(
            "System", 
            "💡 Tip: For best emotion detection, please remove glasses if possible. "
            "Glasses can interfere with facial feature recognition.",
            color="orange"
        )
        
        # Start video thread
        threading.Thread(target=self.update_frame, daemon=True).start()
    
    def _start_webcam(self):
        """Start webcam capture"""
        camera_index = self.detector.config.get("camera_index", 0)
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera. Please check your camera connection.")
            return False
        return True
    
    def stop_detection(self):
        """Stop video capture"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Screen capturer is cleaned up in the update_frame thread
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="⏸️ Stopped")
        self.video_label.configure(image=None, text="Stopped")
    
    def update_frame(self):
        """Update video frame and detect emotions"""
        import time
        
        while self.is_running:
            frame = None
            
            # Get frame from webcam
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    frame = None
            
            if frame is not None:
                    # Detect gestures if enabled
                    if self.gesture_var.get():
                        frame, gesture, gesture_data = self.detector.detect_gestures(frame, show_points=self.show_gesture_points)
                        self.current_gesture = gesture
                        self.current_motion = gesture_data.get("motion", "static")
                        self.update_gesture_display(gesture)
                        
                        # Handle gesture actions
                        self.handle_gesture_action(gesture, self.current_motion)
                    else:
                        self.current_gesture = "disabled"
                        self.current_motion = "static"
                        self.gesture_label.configure(text="👋 Gesture: Disabled")
                    
                    # Apply brightness adjustment
                    frame = self.apply_brightness(frame)
                    
                    # Detect faces
                    faces = self.detector.detect_faces(frame)
                    
                    if len(faces) > 0:
                        # Process first detected face
                        (x, y, w, h) = faces[0]
                        
                        # Get face region
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Predict emotion (with smoothing)
                        emotions = self.detector.predict_emotion(face_roi)
                        
                        # Get dominant emotion
                        dominant = max(emotions.items(), key=lambda x: x[1])
                        self.current_emotion = dominant[0]
                        
                        # Get emotion color
                        emotion_color = self.detector.emotion_colors.get(
                            self.current_emotion, 
                            (0, 255, 0)
                        )
                        
                        # Draw rectangle with emotion color
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_color, 3)
                        
                        # Draw emotion label with background
                        emotion_text = f"{dominant[0].upper()}: {dominant[1]:.2f}"
                        
                        # Get text size for background
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        font_thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            emotion_text, font, font_scale, font_thickness
                        )
                        
                        # Draw background rectangle
                        cv2.rectangle(
                            frame, 
                            (x, y - text_height - 15), 
                            (x + text_width + 10, y - 5),
                            emotion_color, 
                            -1  # Filled
                        )
                        
                        # Draw text in white
                        cv2.putText(
                            frame, emotion_text, (x + 5, y - 10),
                            font, font_scale, (255, 255, 255), font_thickness
                        )
                        
                        # Update UI
                        self.update_emotion_display(emotions, dominant[0])
                    
                    # Convert and display frame
                    self.current_frame = frame
                    self.display_frame(frame)
                    
                    # Auto-analysis check
                    self.check_auto_analysis()
                
            # Control frame rate
            self.after(33)  # ~30 FPS
    
    def display_frame(self, frame):
        """Display video frame in GUI"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        display_width = 640
        display_height = 480
        frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
        
        # Convert to CTkImage
        img = Image.fromarray(frame_resized)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(display_width, display_height))
        
        # Update label
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img  # Keep a reference
    
    def update_smoothing(self, value):
        """Update emotion smoothing window"""
        smoothing_value = int(value)
        self.detector.smoothing_window = smoothing_value
        self.smoothing_label.configure(text=f"{smoothing_value} frames")
    
    def update_gesture_display(self, gesture: str):
        """Update gesture label with emoji, motion, and action description"""
        gesture_emoji_map = {
            "palm_horizontal": "🖐️",
            "open_palm": "🖐️",
            "fist": "✊",
            "peace": "✌️",
            "pointing": "☝️",
            "pinch": "�",
            "four_fingers": "🖖",
            "three_fingers": "🤟",
            "two_fingers": "✌️",
            "unknown": "❓",
            "none": "🚫"
        }
        
        emoji = gesture_emoji_map.get(gesture, "❓")
        gesture_name = gesture.replace("_", " ").title()
        
        # Get action info for this gesture and motion
        action_info = self.detector.get_gesture_action(gesture, self.current_motion)
        action_desc = action_info.get("description", "")
        
        # Build display text
        display_text = f"{emoji} {gesture_name}"
        
        # Add motion indicator
        if self.current_motion != "static":
            motion_arrows = {
                "up": "⬆️",
                "down": "⬇️",
                "left": "⬅️",
                "right": "➡️"
            }
            motion_arrow = motion_arrows.get(self.current_motion, "")
            display_text += f" {motion_arrow}"
        
        # Add action description if available
        if action_desc and action_info.get("action") != "none":
            display_text += f" → {action_desc}"
        
        self.gesture_label.configure(text=display_text)
    
    def apply_brightness(self, frame):
        """Apply brightness adjustment to frame"""
        if self.brightness == 0:
            return frame
        
        # Convert brightness from -100/+100 to -255/+255 scale
        brightness_value = int(self.brightness * 2.55)
        
        # Apply brightness
        if brightness_value > 0:
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_value)
        else:
            frame = cv2.convertScaleAbs(frame, alpha=1 + (brightness_value / 255), beta=0)
        
        # Update slider and label
        self.brightness_slider.set(self.brightness)
        self.brightness_value_label.configure(text=f"{int(self.brightness)}%")
        
        return frame
    
    def set_brightness_manual(self, value):
        """Manually set brightness via slider"""
        self.brightness = int(value)
        self.brightness_value_label.configure(text=f"{int(self.brightness)}%")
    
    def handle_gesture_action(self, gesture: str, motion: str):
        """Handle gesture-triggered actions with cooldown and stability check"""
        import time
        current_time = time.time()
        
        # Track gesture history for stability
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.gesture_stability_frames:
            self.gesture_history.pop(0)
        
        # Check if gesture is stable (same gesture for required frames)
        if len(self.gesture_history) < self.gesture_stability_frames:
            return  # Not enough history yet
        
        # Check if all recent gestures are the same
        if len(set(self.gesture_history)) > 1:
            return  # Gesture is not stable
        
        # Check cooldown
        if current_time - self.last_gesture_action_time < self.gesture_action_cooldown:
            return
        
        # Get action for this gesture
        action_info = self.detector.get_gesture_action(gesture, motion)
        action = action_info.get("action", "none")
        
        if action == "none":
            return
        
        # Execute action
        if action == "brightness_up":
            self.brightness = min(100, self.brightness + 10)
            self.last_gesture_action_time = current_time
            self.gesture_history.clear()  # Clear history after action
            self.status_label.configure(text=f"🔆 Brightness: {self.brightness}%")
            
        elif action == "brightness_down":
            self.brightness = max(-100, self.brightness - 10)
            self.last_gesture_action_time = current_time
            self.gesture_history.clear()
            self.status_label.configure(text=f"🔅 Brightness: {self.brightness}%")
            
        elif action == "take_snapshot":
            self.take_snapshot()
            self.last_gesture_action_time = current_time
            self.gesture_history.clear()
            
        elif action == "clear_chat":
            self.clear_chat()
            self.last_gesture_action_time = current_time
            self.gesture_history.clear()
            
        elif action == "toggle_analysis":
            self.auto_analysis_var.set(not self.auto_analysis_var.get())
            self.toggle_auto_analysis()
            self.last_gesture_action_time = current_time
            self.gesture_history.clear()
            
        elif action == "analyze_frame":
            self.analyze_current_frame()
            self.last_gesture_action_time = current_time
            self.gesture_history.clear()
            
        elif action == "start_voice":
            # Start voice recording with pointing gesture
            if not self.is_voice_recording:
                self.voice_input()
                self.last_gesture_action_time = current_time
                self.gesture_history.clear()
                self.status_label.configure(text="🎤 Voice activated by gesture!")
        
        elif action == "stop_voice":
            # Stop voice recording or stop AI from speaking with call_me gesture
            if self.is_speaking:
                # Stop TTS playback
                if AUDIO_AVAILABLE:
                    pygame.mixer.music.stop()
                self.is_speaking = False
                self.status_label.configure(text="🔇 Voice stopped by gesture!")
                self.add_chat_message("🔇 Voice playback stopped", "system")
            elif self.is_voice_recording:
                self.voice_input()
                self.status_label.configure(text="🔄 Voice processing triggered by gesture!")
            self.last_gesture_action_time = current_time
            self.gesture_history.clear()
        
        elif action == "stop_video":
            # Stop video detection
            if self.is_running:
                self.stop_detection()
                self.last_gesture_action_time = current_time
                self.gesture_history.clear()
        
        elif action == "show_help":
            # Show help dialog
            self.show_help()
            self.last_gesture_action_time = current_time
            self.gesture_history.clear()
    
    def update_emotion_display(self, emotions: dict, dominant_emotion: str):
        """Update emotion bars and labels"""
        # Update progress bars
        for emotion, prob in emotions.items():
            if emotion in self.emotion_bars:
                self.emotion_bars[emotion].set(prob)
        
        # Update dominant emotion label
        emoji_map = {
            "happy": "😊", "sad": "😢", "angry": "😠",
            "fear": "😨", "surprise": "😲", "disgust": "🤢",
            "neutral": "😐"
        }
        
        emoji = emoji_map.get(dominant_emotion, "😐")
        self.current_emotion_label.configure(
            text=f"{emoji} {dominant_emotion}"
        )
    
    def voice_input(self):
        """Toggle voice input capture for the Gemini chat"""
        if not self.detector.recognizer or not SPEECH_RECOGNITION_AVAILABLE:
            self.add_chat_message(
                "❌ Voice input is unavailable on this system.",
                "system",
                color="red"
            )
            return
        
        if not self.is_voice_recording:
            self._start_voice_recording()
        else:
            self._stop_voice_recording()
    
    def _start_voice_recording(self):
        """Begin capturing audio from the microphone"""
        self.stop_speaking()
        if self.voice_thread and self.voice_thread.is_alive():
            return
        
        self.is_voice_recording = True
        self.voice_processing = False
        self.voice_stop_event = threading.Event()
        self.status_label.configure(text="🎤 Listening...")
        
        if hasattr(self, 'voice_btn'):
            self.voice_btn.configure(text="⏹", state="normal")
        
        self.add_chat_message("🎤 Voice recording started... Speak now!", "system", color="blue")
        
        self.voice_thread = threading.Thread(target=self._capture_voice_audio, daemon=True)
        self.voice_thread.start()
    
    def _stop_voice_recording(self):
        """Signal the current recording to stop and start processing"""
        if not self.voice_stop_event:
            return
        
        self.voice_stop_event.set()
        self._prepare_voice_processing()
    
    def _prepare_voice_processing(self):
        """Update UI to indicate voice processing state"""
        if self.voice_processing:
            return
        
        self.voice_processing = True
        self.status_label.configure(text="🔄 Processing voice...")
        if hasattr(self, 'voice_btn'):
            self.voice_btn.configure(state="disabled")
    
    def _capture_voice_audio(self):
        """Background task to record audio frames until stopped"""
        transcript = None
        error_text = None
        frames = []
        sample_rate = None
        sample_width = None
        
        recognizer = self.detector.recognizer
        
        try:
            with sr.Microphone() as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception:
                    pass
                
                sample_rate = getattr(source, "SAMPLE_RATE", 16000)
                sample_width = getattr(source, "SAMPLE_WIDTH", 2)
                
                start_time = time.time()
                while not self.voice_stop_event.is_set():
                    if self.voice_max_duration and (time.time() - start_time) >= self.voice_max_duration:
                        self.voice_stop_event.set()
                        break
                    
                    try:
                        try:
                            chunk = source.stream.read(source.CHUNK, exception_on_overflow=False)
                        except TypeError:
                            chunk = source.stream.read(source.CHUNK)
                        frames.append(chunk)
                    except Exception as exc:
                        print(f"⚠️ Voice capture chunk error: {exc}")
                        break
                
                self.after(0, self._prepare_voice_processing)
        except Exception as exc:
            print(f"❌ Voice capture error: {exc}")
            error_text = "Could not access the microphone. Please check your audio settings."
        
        audio_data = None
        if frames and sample_rate and sample_width and sr:
            audio_data = sr.AudioData(b"".join(frames), sample_rate, sample_width)
        
        if audio_data:
            try:
                transcript = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                transcript = None
            except sr.RequestError as exc:
                print(f"❌ Speech recognition error: {exc}")
                error_text = "Speech recognition service is unavailable right now."
            except Exception as exc:
                print(f"❌ Unexpected speech recognition error: {exc}")
                error_text = "Could not process the recorded audio."
        elif not error_text:
            error_text = "No speech detected or the recording was too short."
        
        self.after(0, lambda: self._finalize_voice_recording(transcript, error_text))
    
    def _finalize_voice_recording(self, transcript: Optional[str], error_text: Optional[str]):
        """Restore UI state after recording and handle transcription output"""
        self.is_voice_recording = False
        self.voice_processing = False
        self.voice_stop_event = None
        self.voice_thread = None
        
        self.status_label.configure(text="Ready")
        if hasattr(self, 'voice_btn'):
            self.voice_btn.configure(text="🎤", state="normal")
        
        if error_text and not transcript:
            self.add_chat_message(f"❌ {error_text}", "system", color="red")
            return
        
        if transcript:
            self.add_chat_message(f"📝 You said: {transcript}", "user")
            self.after(100, lambda: self._send_voice_message_to_ai(transcript))
        else:
            self.add_chat_message("❌ No speech detected or could not understand", "system", color="red")
    
    def _send_voice_message_to_ai(self, message: str):
        """Send voice-transcribed message to AI and get response with TTS"""
        self.stop_speaking()
        def get_response():
            response = self.chatbot.send_message(message, self.current_emotion)
            self.add_chat_message(f"🤖 AI: {response}", "ai")
            
            # Speak the response using ElevenLabs TTS
            if self.tts_enabled and self.elevenlabs_api_key:
                self.speak_text(response)
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def speak_text(self, text: str):
        """Convert text to speech using ElevenLabs API or pyttsx3 fallback"""
        if self.is_speaking:
            return  # Don't overlap speech
        
        self.is_speaking = True
        self.status_label.configure(text="🔊 AI Speaking...")
        
        # Try ElevenLabs first if API key is available
        if self.elevenlabs_api_key and AUDIO_AVAILABLE:
            self._speak_elevenlabs(text)
        # Fallback to pyttsx3
        elif self.tts_engine:
            self._speak_pyttsx3(text)
        else:
            print("⚠️ No TTS engine available")
            self.is_speaking = False
            self.status_label.configure(text="Ready")
    
    def stop_speaking(self):
        """Stop any ongoing TTS playback and reset status"""
        stop_requested = False
        
        if AUDIO_AVAILABLE and pygame.mixer.get_init():
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    stop_requested = True
            except Exception:
                pass
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
                stop_requested = True
            except Exception:
                pass
        
        if self.is_speaking or stop_requested:
            self.is_speaking = False
            self.status_label.configure(text="Ready")
    
    def _speak_elevenlabs(self, text: str):
        """Use ElevenLabs API for TTS"""
        def tts_thread():
            try:
                # Determine which ElevenLabs voice_id to use (preset or custom)
                if hasattr(self, 'eleven_voice_var'):
                    sel = self.eleven_voice_var.get()
                    if sel == 'Custom' and hasattr(self, 'eleven_custom_entry'):
                        custom_id = self.eleven_custom_entry.get().strip()
                        voice_id = custom_id if custom_id else list(self.available_eleven_voices.values())[0]
                    else:
                        voice_id = self.available_eleven_voices.get(sel, list(self.available_eleven_voices.values())[0])
                else:
                    voice_id = list(self.available_eleven_voices.values())[0]

                print(f"🎤 Using ElevenLabs voice: {sel if hasattr(self, 'eleven_voice_var') else 'Rachel'} (ID: {voice_id})")

                # ElevenLabs API endpoint
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": self.elevenlabs_api_key
                }
                
                data = {
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",  # Updated to newer model for free tier
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
                
                # Make API request
                response = requests.post(url, json=data, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    # Save audio to temporary file
                    audio_data = io.BytesIO(response.content)
                    
                    # Play audio using pygame
                    pygame.mixer.music.load(audio_data, 'mp3')
                    pygame.mixer.music.play()
                    
                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    
                    print("✅ ElevenLabs TTS playback complete")
                else:
                    print(f"❌ ElevenLabs API error: {response.status_code}")
                    print(f"Response: {response.text}")
                    print(f"Voice ID used: {voice_id}")
                    # Don't auto-fallback, let user know ElevenLabs failed
                    
            except Exception as e:
                print(f"❌ ElevenLabs TTS error: {e}")
                print(f"Voice ID attempted: {voice_id if 'voice_id' in locals() else 'unknown'}")
            finally:
                self.is_speaking = False
                self.status_label.configure(text="Ready")
        
        threading.Thread(target=tts_thread, daemon=True).start()
    
    def _speak_pyttsx3(self, text: str):
        """Use pyttsx3 for offline TTS (fallback)"""
        def tts_thread():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                print("✅ pyttsx3 TTS playback complete")
            except Exception as e:
                print(f"❌ pyttsx3 TTS error: {e}")
            finally:
                self.is_speaking = False
                self.status_label.configure(text="Ready")
        
        threading.Thread(target=tts_thread, daemon=True).start()
    
    def send_message(self):
        """Send message to AI chatbot"""
        self.stop_speaking()
        message = self.chat_input.get().strip()
        
        if not message:
            return
        
        # Display user message
        self.add_chat_message(f"You: {message}", "user")
        
        # Clear input
        self.chat_input.delete(0, "end")
        
        # Get AI response
        def get_response():
            response = self.chatbot.send_message(message, self.current_emotion)
            self.add_chat_message(f"AI: {response}", "ai")
            
            if self.tts_enabled and (self.elevenlabs_api_key or self.tts_engine):
                self.speak_text(response)
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def add_chat_message(self, message: str, sender: str, color: Optional[str] = None):
        """Add message to chat display with optional color"""
        self.chat_display.configure(state="normal")
        
        timestamp = datetime.now().strftime("%H:%M")
        
        # Configure text tags with colors if not already configured
        try:
            self.chat_display.tag_config("orange", foreground="#ff8800")
            self.chat_display.tag_config("green", foreground="#00ff00")
            self.chat_display.tag_config("red", foreground="#ff0000")
            self.chat_display.tag_config("blue", foreground="#00aaff")
        except:
            pass
        
        if sender == "user":
            self.chat_display.insert("end", f"[{timestamp}] {message}\n\n", "user")
        elif sender == "ai":
            self.chat_display.insert("end", f"[{timestamp}] {message}\n\n", "ai")
        elif color:
            self.chat_display.insert("end", f"[{timestamp}] {message}\n\n", color)
        else:
            self.chat_display.insert("end", f"[{timestamp}] {message}\n\n")
        
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self.chatbot.clear_history()
    
    def change_personality(self, personality: str):
        """Change AI personality"""
        self.chatbot.set_personality(personality)
        self.add_chat_message(f"Personality changed to: {personality}", "system")
    
    def take_snapshot(self):
        """Save current frame as image"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame to capture. Start detection first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = EXPORTS_DIR / f"snapshot_{timestamp}.png"
        
        cv2.imwrite(str(filename), self.current_frame)
        
        self.status_label.configure(text=f"📸 Saved: {filename.name}")
        messagebox.showinfo("Success", f"Snapshot saved to:\n{filename}")
    
    def upload_image(self):
        """Upload and analyze an image for emotions and with Gemini AI"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if filename:
            try:
                # Load the image
                image = cv2.imread(filename)
                if image is None:
                    messagebox.showerror("Error", "Could not load image file.")
                    return
                
                # Store for image mode
                self.uploaded_image = image.copy()
                
                # Analyze for emotions
                self._analyze_static_image(image)
                
                # Also analyze with Gemini AI in background
                def gemini_analysis():
                    try:
                        response = self.chatbot.analyze_image(filename)
                        self.add_chat_message("AI", f"🤖 Image Analysis:\n{response}", color="purple")
                    except Exception as e:
                        self.add_chat_message("System", f"AI analysis error: {str(e)}", color="red")
                
                threading.Thread(target=gemini_analysis, daemon=True).start()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def _analyze_static_image(self, image):
        """Analyze a static image for face and emotions"""
        try:
            # Make a copy for annotation
            annotated_frame = image.copy()
            
            # Detect faces
            faces = self.detector.detect_faces(image)
            
            if len(faces) > 0:
                all_emotions = []
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Get face region
                    face_roi = image[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotions = self.detector.predict_emotion(face_roi)
                    all_emotions.append(emotions)
                    
                    # Get dominant emotion
                    dominant = max(emotions.items(), key=lambda x: x[1])
                    
                    # Get emotion color
                    emotion_color = self.detector.emotion_colors.get(
                        dominant[0], 
                        (0, 255, 0)
                    )
                    
                    # Draw rectangle with emotion color
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), emotion_color, 3)
                    
                    # Draw emotion label with background
                    emotion_text = f"{dominant[0].upper()}: {dominant[1]:.1f}%"
                    
                    # Get text size for background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(
                        emotion_text, font, font_scale, font_thickness
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        annotated_frame, 
                        (x, y - text_height - 15), 
                        (x + text_width + 10, y - 5),
                        emotion_color, 
                        -1  # Filled
                    )
                    
                    # Draw text in white
                    cv2.putText(
                        annotated_frame, emotion_text, (x + 5, y - 10),
                        font, font_scale, (255, 255, 255), font_thickness
                    )
                
                # Update display
                self.current_frame = annotated_frame
                self.display_frame(annotated_frame)
                
                # Update emotion bars with first face's emotions
                face_emotions = all_emotions[0]
                for emotion, value in face_emotions.items():
                    if emotion in self.emotion_bars:
                        self.emotion_bars[emotion].set(value / 100.0)  # Convert to 0-1 range
                
                # Get dominant emotion from first face
                dominant_emotion = max(face_emotions.items(), key=lambda x: x[1])
                self.current_emotion = dominant_emotion[0]
                
                # Show result
                self.add_chat_message(
                    "System",
                    f"✅ Detected {len(faces)} face(s). Dominant emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.1f}%)",
                    color="green"
                )
            else:
                # No faces detected - still display the image
                self.current_frame = annotated_frame
                self.display_frame(annotated_frame)
                
                self.add_chat_message(
                    "System",
                    "⚠️ No faces detected in the image.",
                    color="orange"
                )
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to analyze image: {str(e)}")
    
    def analyze_current_frame(self):
        """Analyze the current webcam frame with Gemini AI"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available. Start detection first.")
            return
        
        self.status_label.configure(text="🤖 AI analyzing frame...")
        self.ai_vision_btn.configure(state="disabled")
        
        def analyze():
            try:
                # Analyze the current frame
                response = self.chatbot.analyze_frame(
                    self.current_frame,
                    f"Describe what you see in this image. Focus on the person's facial expression, "
                    f"emotion, and overall appearance. The detected emotion is: {self.current_emotion}"
                )
                
                # Add to chat
                self.add_chat_message(f"📹 Webcam Analysis:\n{response}", "ai")
                self.status_label.configure(text="✅ Analysis complete")
                
            except Exception as e:
                error_msg = f"Error analyzing frame: {str(e)}"
                self.add_chat_message(error_msg, "system")
                self.status_label.configure(text="❌ Analysis failed")
            
            finally:
                self.ai_vision_btn.configure(state="normal")
        
        # Run in background thread
        threading.Thread(target=analyze, daemon=True).start()
    
    def toggle_gesture_visibility(self):
        """Show/hide gesture points button when gesture detection is toggled"""
        if self.gesture_var.get():
            # Gestures enabled - show points button
            self.gesture_points_btn.pack(side="left", padx=5)
        else:
            # Gestures disabled - hide points button and reset points display
            self.gesture_points_btn.pack_forget()
            self.show_gesture_points = False
    
    def toggle_gesture_points(self):
        """Toggle display of hand landmark points"""
        self.show_gesture_points = not self.show_gesture_points
        if self.show_gesture_points:
            self.gesture_points_btn.configure(fg_color="#10b981", hover_color="#059669")
            self.add_chat_message("📍 Gesture points enabled", "system", color="green")
        else:
            self.gesture_points_btn.configure(fg_color="#7c3aed", hover_color="#6d28d9")
            self.add_chat_message("📍 Gesture points disabled", "system")

    def change_voice(self, selection: str):
        """Handle unified voice selection (ElevenLabs or Offline)."""
        try:
            if selection.startswith("[EL]"):
                # ElevenLabs voice selected
                voice_name = selection.replace("[EL] ", "")
                
                if voice_name == "Custom":
                    # Show custom entry field
                    if not self.eleven_custom_entry.winfo_ismapped():
                        self.eleven_custom_entry.pack(side="left", padx=(5, 0))
                else:
                    # Hide custom entry field
                    if self.eleven_custom_entry.winfo_ismapped():
                        self.eleven_custom_entry.pack_forget()
                
                # Save selection
                self.eleven_voice_name = voice_name
                self.add_chat_message(f"🎤 Voice set to: {voice_name} (ElevenLabs)", "system", color="green")
                
            elif selection.startswith("[Offline]"):
                # Offline pyttsx3 voice selected
                voice_name = selection.replace("[Offline] ", "")
                
                # Hide custom entry if visible
                if self.eleven_custom_entry.winfo_ismapped():
                    self.eleven_custom_entry.pack_forget()
                
                # Change pyttsx3 voice
                voice_id = self.pyttsx3_voice_map.get(voice_name)
                if voice_id and self.tts_engine:
                    try:
                        self.tts_engine.setProperty('voice', voice_id)
                        self.pyttsx3_selected_voice = voice_name
                        self.add_chat_message(f"🔊 Voice changed to: {voice_name} (Offline)", "system", color="green")
                    except Exception as e:
                        self.add_chat_message(f"❌ Could not set offline voice: {e}", "system", color="red")
        except Exception as e:
            print(f"Error changing voice: {e}")
    
    def check_auto_analysis(self):
        """Check if it's time to run auto-analysis"""
        if not self.auto_analysis_enabled or self.current_frame is None:
            return
        
        import time
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_analysis_time >= self.analysis_interval:
            self.last_analysis_time = current_time
            
            # Run analysis in background
            def auto_analyze():
                try:
                    response = self.chatbot.analyze_frame(
                        self.current_frame,
                        f"Briefly describe the person's current state and emotion. Detected: {self.current_emotion}"
                    )
                    self.add_chat_message(f"🤖 Auto Analysis:\n{response}", "ai")
                except Exception as e:
                    print(f"Auto-analysis error: {e}")
            
            threading.Thread(target=auto_analyze, daemon=True).start()
    
    def test_all_voices(self):
        """Test currently selected voice (ElevenLabs or Offline)"""
        self.test_voices_btn.configure(state="disabled", text="Testing...")
        
        def test_voice():
            try:
                # Get currently selected voice from unified selector
                selected = self.unified_voice_var.get()
                test_text = "Hello! This is a test of the text to speech system."
                
                if selected.startswith("[EL]"):
                    # Test ElevenLabs voice
                    voice_name = selected.replace("[EL] ", "")
                    
                    if not self.elevenlabs_api_key:
                        self.add_chat_message("❌ No ElevenLabs API key configured", "system", color="red")
                        self.test_voices_btn.configure(state="normal", text="🎤 Test Voice")
                        return
                    
                    if not AUDIO_AVAILABLE:
                        self.add_chat_message("❌ Audio playback not available (install pygame)", "system", color="red")
                        self.test_voices_btn.configure(state="normal", text="🎤 Test Voice")
                        return
                    
                    # Get voice ID
                    if voice_name == 'Custom' and hasattr(self, 'eleven_custom_entry'):
                        custom_id = self.eleven_custom_entry.get().strip()
                        voice_id = custom_id if custom_id else list(self.available_eleven_voices.values())[0]
                    else:
                        voice_id = self.available_eleven_voices.get(voice_name, list(self.available_eleven_voices.values())[0])
                    
                    self.add_chat_message(f"🎤 Testing ElevenLabs voice: {voice_name}...", "system", color="blue")
                    
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                    
                    headers = {
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                        "xi-api-key": self.elevenlabs_api_key
                    }
                    
                    data = {
                        "text": test_text,
                        "model_id": "eleven_turbo_v2_5",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75
                        }
                    }
                    
                    response = requests.post(url, json=data, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        # Play audio
                        audio_data = io.BytesIO(response.content)
                        pygame.mixer.music.load(audio_data, 'mp3')
                        pygame.mixer.music.play()
                        
                        # Wait for playback to finish
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        
                        self.add_chat_message(f"✅ {voice_name} - Working! ({len(response.content)} bytes)", "system", color="green")
                    else:
                        self.add_chat_message(f"❌ {voice_name} - Failed ({response.status_code})", "system", color="red")
                        self.add_chat_message(f"Response: {response.text}", "system", color="red")
                
                elif selected.startswith("[Offline]"):
                    # Test Offline voice
                    voice_name = selected.replace("[Offline] ", "")
                    
                    if not self.tts_engine:
                        self.add_chat_message("❌ Offline TTS engine not available", "system", color="red")
                        self.test_voices_btn.configure(state="normal", text="🎤 Test Voice")
                        return
                    
                    self.add_chat_message(f"🔊 Testing Offline voice: {voice_name}...", "system", color="blue")
                    
                    # Get voice ID
                    voice_id = self.pyttsx3_voice_map.get(voice_name)
                    if voice_id:
                        try:
                            self.tts_engine.setProperty('voice', voice_id)
                            self.tts_engine.say(test_text)
                            self.tts_engine.runAndWait()
                            self.add_chat_message(f"✅ {voice_name} - Working!", "system", color="green")
                        except Exception as e:
                            self.add_chat_message(f"❌ {voice_name} - Failed: {str(e)}", "system", color="red")
                    else:
                        self.add_chat_message(f"❌ Voice not found: {voice_name}", "system", color="red")
                else:
                    self.add_chat_message("❌ No voice selected", "system", color="red")
                
            except Exception as e:
                self.add_chat_message(f"❌ Test Error: {str(e)}", "system", color="red")
            
            finally:
                self.test_voices_btn.configure(state="normal", text="🎤 Test Voice")
        
        threading.Thread(target=test_voice, daemon=True).start()
    
    def test_chatbot(self):
        """Test the Gemini chatbot functionality and show model info"""
        self.test_chatbot_btn.configure(state="disabled", text="Testing...")
        
        def run_test():
            try:
                # Get model info
                self.add_chat_message("🤖 Testing Gemini Chatbot...", "system", color="blue")
                
                info = self.chatbot.get_model_info()
                
                # Display model information
                self.add_chat_message(
                    f"📊 Chatbot Status:\n"
                    f"• Model: {info['current_model']}\n"
                    f"• API Configured: {info['api_configured']}\n"
                    f"• Available Models: {info['total_available']}",
                    "system",
                    color="cyan"
                )
                
                # Show top 3 available models
                if info['available_models']:
                    models_list = "\n".join([f"  {i+1}. {m}" for i, m in enumerate(info['available_models'][:3])])
                    self.add_chat_message(
                        f"🎯 Top Available Models:\n{models_list}",
                        "system",
                        color="cyan"
                    )
                
                # Test a conversation
                test_response = self.chatbot.send_message("Hello! How are you today?", "happy")
                self.add_chat_message("Test message sent: 'Hello! How are you today?'", "user")
                self.add_chat_message(test_response, "ai")
                
                # Get conversation summary
                summary = self.chatbot.get_conversation_summary()
                self.add_chat_message(
                    f"📈 Conversation Summary:\n"
                    f"• Total Messages: {summary['total_messages']}\n"
                    f"• Personality: {summary['personality']}\n"
                    f"• Emotions Discussed: {', '.join(summary['emotions_discussed'])}",
                    "system",
                    color="green"
                )
                
                self.add_chat_message("✅ Chatbot test complete!", "system", color="green")
                
            except Exception as e:
                self.add_chat_message(f"❌ Chatbot test failed: {str(e)}", "system", color="red")
                import traceback
                traceback.print_exc()
            
            finally:
                self.test_chatbot_btn.configure(state="normal", text="🤖 Test Chat")
        
        threading.Thread(target=run_test, daemon=True).start()
    
    def save_session(self):
        """Save current session data"""
        filename = self.detector.save_session()
        messagebox.showinfo("Success", f"Session saved to:\n{filename}")
    
    def export_csv(self):
        """Export session data to CSV"""
        filename = self.detector.export_csv()
        if filename:
            messagebox.showinfo("Success", f"Data exported to:\n{filename}")
    
    def show_help(self):
        """Show comprehensive help dialog"""
        help_window = ctk.CTkToplevel(self)
        help_window.title("📚 Emotion Detection System - Help Guide")
        help_window.geometry("900x700")
        
        # Make window modal
        help_window.grab_set()
        
        # Title
        title_label = ctk.CTkLabel(
            help_window, 
            text="📚 Complete Feature Guide",
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=20)
        
        # Scrollable help content
        help_scroll = ctk.CTkScrollableFrame(help_window, width=850, height=550)
        help_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Important tips at the top
        tips_frame = ctk.CTkFrame(help_scroll, fg_color="#ff8800", corner_radius=10)
        tips_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(
            tips_frame,
            text="💡 IMPORTANT TIPS FOR BEST RESULTS",
            font=("Arial", 16, "bold"),
            text_color="white"
        ).pack(pady=(10, 5), padx=15, anchor="w")
        
        tips = [
            "👓 Remove glasses for better emotion detection accuracy",
            "💡 Ensure good lighting on your face",
            "📏 Keep face centered and at comfortable distance from camera",
            "👋 Hold gestures steady for 5 frames (about 0.5 seconds) to activate",
            "🤏 For brightness: bring fingers together = UP, move apart = DOWN",
            "☝️ Use Pointing gesture (index finger) to START voice chat",
            "✊ Use Fist gesture (closed hand) to STOP voice/speaking",
        ]
        
        for tip in tips:
            ctk.CTkLabel(
                tips_frame,
                text=f"  • {tip}",
                font=("Arial", 11),
                text_color="white",
                anchor="w"
            ).pack(pady=2, padx=20, anchor="w")
        
        ctk.CTkLabel(tips_frame, text="", height=5).pack()
        
        # Help content with sections
        help_sections = [
            {
                "title": "🎮 Control Buttons",
                "items": [
                    ("▶️ Start", "Start camera and emotion detection"),
                    ("⏹️ Stop", "Stop camera and detection"),
                    ("📸 Snapshot", "Save current frame as image to data/exports/"),
                    ("🖼️ Upload Image", "Upload and analyze a photo with emotion detection + AI"),
                    ("🤖 AI Vision", "Analyze current frame with Gemini AI"),
                    ("👋 Gestures", "Toggle hand gesture detection on/off"),
                    ("⚙️ Settings", "Configure API keys"),
                    ("❓ Help", "Show this help guide"),
                    ("💾 Save", "Save current session data to JSON"),
                    ("📄 Export CSV", "Export emotion data to CSV file"),
                ]
            },
            {
                "title": "😊 Emotion Colors",
                "items": [
                    ("🔴 Red Box", "ANGRY - Aggressive, intense emotion"),
                    ("🟢 Green Box", "DISGUST - Unpleasant, repulsive feeling"),
                    ("🟣 Purple Box", "FEAR - Anxious, worried state"),
                    ("🟡 Yellow Box", "HAPPY - Joyful, positive emotion"),
                    ("⚪ Gray Box", "NEUTRAL - Calm, balanced state"),
                    ("🟠 Orange Box", "SAD - Down, melancholic feeling"),
                    ("🔵 Cyan Box", "SURPRISE - Shocked, amazed reaction"),
                ]
            },
            {
                "title": "👋 Hand Gestures Guide",
                "items": [
                    ("☝️ Pointing (Index only)", "🎤 START VOICE CHAT - Activate voice recording!"),
                    ("✊ Fist (Closed hand)", "🔇 STOP VOICE - Stop voice recording or AI speaking"),
                    ("✌️ Peace Sign (Index+Middle)", "Stops video detection"),
                    ("🤏 Pinch (Thumb+Index)", "Pinch motion controls brightness"),
                ]
            },
            {
                "title": "🎮 Hand Motion Controls",
                "items": [
                    ("🤏 Pinch Gesture", "Bring thumb and index finger together/apart"),
                    ("⬆️ Fingers TOGETHER", "Increases brightness +10%"),
                    ("⬇️ Fingers APART", "Decreases brightness -10%"),
                    ("How It Works", "Move fingers closer = UP, move apart = DOWN"),
                    ("Motion Range", "Brightness: -100% to +100%"),
                    ("Stability", "Hold gesture for 5 frames to activate"),
                    ("Cooldown", "2 seconds between gesture actions"),
                ]
            },
            {
                "title": "🔆 Brightness Control",
                "items": [
                    ("Slider", "Manually adjust brightness -100% to +100%"),
                    ("Gesture Control", "Move hand up/down while showing any gesture"),
                    ("Real-time", "Video brightness updates immediately"),
                    ("Reset", "Drag slider to 0% to reset"),
                ]
            },
            {
                "title": "📊 Analysis Panel",
                "items": [
                    ("Colored Bars", "Each emotion has its own color"),
                    ("Current Emotion", "Large emoji shows dominant emotion"),
                    ("Smoothing Slider", "Adjust volatility (1-10 frames)"),
                    ("🎤 Voice Input", "Speak to analyze your voice (if available)"),
                ]
            },
            {
                "title": "💬 AI Chat Features",
                "items": [
                    ("Personality", "Choose: Empathetic, Neutral, or Professional"),
                    ("Context-Aware", "AI knows your current detected emotion"),
                    ("Image Analysis", "Upload images for AI description"),
                    ("🎤 Voice Chat", "Use ☝️ Pointing gesture to START voice recording"),
                    ("🔇 Stop Voice", "Use ✊ Fist gesture to STOP voice/speaking"),
                    ("Voice Recording", "Speak for up to 10 seconds, AI responds automatically"),
                    ("Clear Chat", "Removes all conversation history"),
                ]
            },
            {
                "title": "🎯 Tips for Best Results",
                "items": [
                    ("Face Detection", "Look directly at camera with good lighting"),
                    ("With Glasses", "Ensure front lighting, avoid glare"),
                    ("Gestures", "Show full hand, palm facing camera"),
                    ("Smoothing", "Higher value = more stable, less reactive"),
                    ("Motion", "Smooth, moderate-speed up/down movements"),
                ]
            },
            {
                "title": "⌨️ Keyboard Shortcuts",
                "items": [
                    ("Enter (in chat)", "Send message to AI"),
                    ("Slider Drag", "Adjust smoothing or brightness"),
                    ("Toggle Switches", "Click to enable/disable features"),
                ]
            },
            {
                "title": "💾 Data Export",
                "items": [
                    ("Snapshots", "Saved to: data/exports/snapshot_[timestamp].png"),
                    ("Sessions", "Saved to: data/sessions/session_[timestamp].json"),
                    ("CSV Export", "Emotion data in spreadsheet format"),
                    ("Conversations", "AI chat history with emotions"),
                ]
            },
        ]
        
        # Create sections
        for section in help_sections:
            # Section header
            section_frame = ctk.CTkFrame(help_scroll, fg_color="#2b2b2b", corner_radius=10)
            section_frame.pack(fill="x", pady=10, padx=10)
            
            header = ctk.CTkLabel(
                section_frame,
                text=section["title"],
                font=("Arial", 16, "bold"),
                anchor="w"
            )
            header.pack(pady=10, padx=15, anchor="w")
            
            # Section items
            for item, description in section["items"]:
                item_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
                item_frame.pack(fill="x", padx=20, pady=2)
                
                item_label = ctk.CTkLabel(
                    item_frame,
                    text=f"  • {item}:",
                    font=("Arial", 12, "bold"),
                    anchor="w",
                    width=200
                )
                item_label.pack(side="left", padx=(0, 10))
                
                desc_label = ctk.CTkLabel(
                    item_frame,
                    text=description,
                    font=("Arial", 11),
                    anchor="w",
                    wraplength=550
                )
                desc_label.pack(side="left", fill="x", expand=True)
            
            # Add spacing
            ctk.CTkLabel(section_frame, text="", height=5).pack()
        
        # Close button
        close_btn = ctk.CTkButton(
            help_window,
            text="Close",
            command=help_window.destroy,
            width=200,
            height=40,
            font=("Arial", 14, "bold")
        )
        close_btn.pack(pady=10)
    
    def show_settings(self):
        """Show settings dialog for API keys and configuration"""
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("⚙️ Settings")
        settings_window.geometry("700x600")
        
        # Make window modal
        settings_window.grab_set()
        
        # Title
        title_label = ctk.CTkLabel(
            settings_window, 
            text="⚙️ Application Settings",
            font=("Arial", 24, "bold")
        )
        title_label.pack(pady=20)
        
        # Scrollable settings content
        settings_scroll = ctk.CTkScrollableFrame(settings_window, width=650, height=420)
        settings_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        # === API Keys Section ===
        api_frame = ctk.CTkFrame(settings_scroll, fg_color="#2b2b2b", corner_radius=10)
        api_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(
            api_frame,
            text="🔑 API Keys",
            font=("Arial", 18, "bold")
        ).pack(pady=10, padx=15, anchor="w")
        
        # Gemini API Key
        ctk.CTkLabel(
            api_frame,
            text="Google Gemini API Key:",
            font=("Arial", 12, "bold"),
            anchor="w"
        ).pack(pady=(10, 2), padx=20, anchor="w")
        
        ctk.CTkLabel(
            api_frame,
            text="Required for AI Vision and chatbot features",
            font=("Arial", 10),
            text_color="gray",
            anchor="w"
        ).pack(pady=(0, 5), padx=20, anchor="w")
        
        self.gemini_key_entry = ctk.CTkEntry(
            api_frame,
            width=600,
            placeholder_text="Enter your Gemini API key (get from https://makersuite.google.com/app/apikey)",
            show="•"  # Hide the key
        )
        self.gemini_key_entry.pack(pady=5, padx=20)
        
        # Load current value from config or chatbot
        current_gemini_key = self.detector.config.get("gemini_api_key", "")
        if not current_gemini_key or current_gemini_key == "YOUR_GEMINI_API_KEY_HERE":
            # Fallback to chatbot api_key if config is empty
            current_gemini_key = self.chatbot.api_key if hasattr(self.chatbot, 'api_key') else ""
        
        if current_gemini_key and current_gemini_key != "YOUR_GEMINI_API_KEY_HERE":
            self.gemini_key_entry.insert(0, current_gemini_key)
        
        # Show/Hide Gemini key button
        show_gemini_var = ctk.BooleanVar(value=False)
        def toggle_gemini_key():
            self.gemini_key_entry.configure(show="" if show_gemini_var.get() else "•")
        
        ctk.CTkCheckBox(
            api_frame,
            text="Show API Key",
            variable=show_gemini_var,
            command=toggle_gemini_key,
            font=("Arial", 10)
        ).pack(pady=2, padx=20, anchor="w")
        
        # ElevenLabs API Key
        ctk.CTkLabel(
            api_frame,
            text="ElevenLabs API Key (Optional):",
            font=("Arial", 12, "bold"),
            anchor="w"
        ).pack(pady=(15, 2), padx=20, anchor="w")
        
        ctk.CTkLabel(
            api_frame,
            text="Optional - for premium text-to-speech. Falls back to offline voice if not provided.",
            font=("Arial", 10),
            text_color="gray",
            anchor="w"
        ).pack(pady=(0, 5), padx=20, anchor="w")
        
        self.elevenlabs_key_entry = ctk.CTkEntry(
            api_frame,
            width=600,
            placeholder_text="Enter your ElevenLabs API key (optional, get from https://elevenlabs.io)",
            show="•"
        )
        self.elevenlabs_key_entry.pack(pady=5, padx=20)
        
        # Load current value
        if self.elevenlabs_api_key and self.elevenlabs_api_key.strip():
            self.elevenlabs_key_entry.insert(0, self.elevenlabs_api_key)
        
        # Show/Hide ElevenLabs key button
        show_eleven_var = ctk.BooleanVar(value=False)
        def toggle_eleven_key():
            self.elevenlabs_key_entry.configure(show="" if show_eleven_var.get() else "•")
        
        ctk.CTkCheckBox(
            api_frame,
            text="Show API Key",
            variable=show_eleven_var,
            command=toggle_eleven_key,
            font=("Arial", 10)
        ).pack(pady=2, padx=20, anchor="w")
        
        ctk.CTkLabel(api_frame, text="", height=10).pack()
        
        # === Buttons ===
        button_frame = ctk.CTkFrame(settings_window, fg_color="transparent")
        button_frame.pack(pady=10)
        
        # Save button
        save_btn = ctk.CTkButton(
            button_frame,
            text="💾 Save Settings",
            command=lambda: self.save_settings(settings_window),
            width=150,
            height=40,
            font=("Arial", 14, "bold"),
            fg_color="#059669",
            hover_color="#047857"
        )
        save_btn.pack(side="left", padx=10)
        
        # Cancel button
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="❌ Cancel",
            command=settings_window.destroy,
            width=150,
            height=40,
            font=("Arial", 14, "bold"),
            fg_color="#dc2626",
            hover_color="#b91c1c"
        )
        cancel_btn.pack(side="left", padx=10)
    
    def save_settings(self, settings_window):
        """Save settings to config file and apply changes"""
        try:
            # Get values from entries
            gemini_key = self.gemini_key_entry.get().strip()
            elevenlabs_key = self.elevenlabs_key_entry.get().strip()
            
            # Update API keys - always save, even if empty
            self.detector.config["gemini_api_key"] = gemini_key if gemini_key else ""
            self.detector.config["elevenlabs_api_key"] = elevenlabs_key if elevenlabs_key else ""
            
            # Update chatbot with Gemini key
            if gemini_key:
                self.chatbot.api_key = gemini_key
                # Reinitialize chatbot with new key
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key)
                    self.chatbot.model = genai.GenerativeModel(
                        model_name='gemini-1.5-flash',
                        generation_config=self.chatbot.generation_config,
                        safety_settings=self.chatbot.safety_settings
                    )
                    self.chatbot.chat = self.chatbot.model.start_chat(history=[])
                except Exception as e:
                    print(f"Error updating Gemini API: {e}")
            
            # Update ElevenLabs key
            if elevenlabs_key:
                self.elevenlabs_api_key = elevenlabs_key
            
            # Save to config file
            config_path = PROJECT_ROOT / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.detector.config, f, indent=4)
            
            # Show success message
            messagebox.showinfo(
                "Settings Saved",
                "Settings have been saved successfully!\n\n"
                "Changes will take effect immediately for:\n"
                "• API keys\n\n"
                "Camera index will apply on next detection start."
            )
            
            # Add to chat
            self.add_chat_message(
                "System",
                "⚙️ Settings updated successfully!",
                color="green"
            )
            
            # Close settings window
            settings_window.destroy()
            
        except Exception as e:
            messagebox.showerror(
                "Error Saving Settings",
                f"Failed to save settings:\n{str(e)}"
            )
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.destroy()


def main():
    """Main entry point"""
    app = EmotionDetectionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
