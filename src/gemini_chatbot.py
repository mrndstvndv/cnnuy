"""
Gemini AI Chatbot Module
Provides emotion-aware AI conversations
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Get project root directory (parent of src)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

# Ensure directories exist
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Generative AI not available - chatbot features disabled")


class GeminiChatbot:
    """AI chatbot powered by Google's Gemini with emotion awareness"""
    
    def __init__(self, config_path: str = "../config.json"):
        """Initialize the Gemini chatbot"""
        self.config = self._load_config(config_path)
        self.conversation_history = []
        self.current_emotion = "neutral"
        self.personality = self.config.get("ai_personality", "empathetic")
        self.available_models = []
        
        # Initialize Gemini
        if GEMINI_AVAILABLE:
            self._initialize_gemini()
        else:
            self.model = None
            self.chat = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load config: {e}")
            return {}
    
    def _list_available_models(self) -> List[str]:
        """List all available Gemini models that support generateContent"""
        try:
            models = genai.list_models()
            available = []
            
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    # Filter for main Gemini models (exclude specialized ones)
                    model_name = model.name.replace('models/', '')
                    if 'gemini' in model_name.lower() and not any(x in model_name.lower() for x in ['embedding', 'vision', 'live', 'thinking', 'tts', 'robotics', 'computer-use']):
                        available.append(model_name)
            
            return available
        except Exception as e:
            print(f"⚠️ Could not list models: {e}")
            return []
    
    def _select_best_model(self, available_models: List[str]) -> Optional[str]:
        """Select the best available model from the list"""
        # Preference order: stable flash > stable pro > preview flash > other
        preference_order = [
            'gemini-2.5-flash',
            'gemini-flash-latest',
            'gemini-2.0-flash',
            'gemini-2.5-pro',
            'gemini-pro-latest',
            'gemini-2.0-pro-exp',
        ]
        
        # Check preferred models first
        for preferred in preference_order:
            if preferred in available_models:
                return preferred
        
        # Fallback: return first flash model
        for model in available_models:
            if 'flash' in model.lower():
                return model
        
        # Last resort: return any available model
        return available_models[0] if available_models else None
    
    def _initialize_gemini(self):
        """Initialize Gemini AI model with automatic model detection"""
        api_key = self.config.get("gemini_api_key", "")
        
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            print("⚠️ Gemini API key not configured")
            self.model = None
            self.chat = None
            return
        
        try:
            genai.configure(api_key=api_key)
            
            # List available models
            self.available_models = self._list_available_models()
            
            if not self.available_models:
                print("⚠️ No compatible Gemini models found")
                self.model = None
                self.chat = None
                return
            
            # Select best model
            model_name = self._select_best_model(self.available_models)
            
            if not model_name:
                print("⚠️ Could not select a suitable model")
                self.model = None
                self.chat = None
                return
            
            # Initialize model
            self.model = genai.GenerativeModel(model_name)
            self.chat = self.model.start_chat(history=[])
            print(f"✅ Gemini AI initialized successfully with model: {model_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            self.model = None
            self.chat = None
    
    def _get_personality_prompt(self) -> str:
        """Get system prompt based on personality mode"""
        base_prompt = "You are a helpful AI assistant. "
        
        if self.personality == "empathetic":
            return (base_prompt + 
                   "You are warm, understanding, and emotionally supportive. "
                   "You acknowledge the user's feelings and provide compassionate responses. "
                   "When the user seems upset, you offer comfort. When they're happy, you share their joy.")
        
        elif self.personality == "professional":
            return (base_prompt +
                   "You are professional, concise, and focused on providing clear information. "
                   "You maintain a respectful and businesslike tone while still being helpful.")
        
        else:  # neutral
            return (base_prompt +
                   "You provide balanced, informative responses without excessive emotional engagement.")
    
    def _get_emotion_context(self, emotion: str) -> str:
        """Generate context based on detected emotion"""
        emotion_contexts = {
            "happy": f"IMPORTANT: The emotion detector currently shows the user's highest detected emotion is HAPPY. The user seems happy and positive right now.",
            "sad": f"IMPORTANT: The emotion detector currently shows the user's highest detected emotion is SAD. The user appears to be feeling sad or down at the moment.",
            "angry": f"IMPORTANT: The emotion detector currently shows the user's highest detected emotion is ANGRY. The user seems frustrated or angry currently.",
            "fear": f"IMPORTANT: The emotion detector currently shows the user's highest detected emotion is FEAR. The user appears anxious or worried.",
            "surprise": f"IMPORTANT: The emotion detector currently shows the user's highest detected emotion is SURPRISE. The user seems surprised or startled.",
            "disgust": f"IMPORTANT: The emotion detector currently shows the user's highest detected emotion is DISGUST. The user appears displeased or disgusted.",
            "neutral": f"IMPORTANT: The emotion detector currently shows the user's highest detected emotion is NEUTRAL. The user has a neutral emotional state."
        }
        
        return emotion_contexts.get(emotion, f"IMPORTANT: The emotion detector currently shows the user's highest detected emotion is {emotion.upper()}.")
    
    def send_message(self, message: str, detected_emotion: Optional[str] = None) -> str:
        """
        Send a message to Gemini and get a response
        
        Args:
            message: User's message
            detected_emotion: Currently detected emotion (optional)
            
        Returns:
            AI response text
        """
        if not GEMINI_AVAILABLE or self.model is None:
            return "⚠️ Gemini AI is not available. Please check your API key configuration."
        
        try:
            # Always update current emotion if provided (use latest detected emotion)
            if detected_emotion:
                self.current_emotion = detected_emotion
            
            # Build context-aware prompt - ALWAYS include current emotion
            personality_prompt = self._get_personality_prompt()
            emotion_context = self._get_emotion_context(self.current_emotion)
            
            # Always include emotion context so chatbot knows current emotional state
            # Include personality and emotion context with every message
            full_message = f"{personality_prompt}\n{emotion_context}\n\nUser: {message}"
            
            # Send message and get response
            response = self.chat.send_message(full_message)
            response_text = response.text
            
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "ai_response": response_text,
                "detected_emotion": self.current_emotion
            })
            
            return response_text
            
        except Exception as e:
            error_msg = f"❌ Error communicating with Gemini: {str(e)}"
            print(error_msg)
            return error_msg
    
    def analyze_image(self, image_path: str, question: str = "What do you see in this image?") -> str:
        """
        Analyze an image using Gemini Vision
        
        Args:
            image_path: Path to the image file
            question: Question about the image
            
        Returns:
            AI description/analysis of the image
        """
        if not GEMINI_AVAILABLE or self.model is None:
            return "⚠️ Gemini AI is not available."
        
        try:
            # Use the same model (modern Gemini models support vision)
            vision_model = self.model
            
            # Read image file
            import PIL.Image
            img = PIL.Image.open(image_path)
            
            # Generate response
            response = vision_model.generate_content([question, img])
            
            return response.text
            
        except Exception as e:
            error_msg = f"❌ Error analyzing image: {str(e)}"
            print(error_msg)
            return error_msg
    
    def analyze_frame(self, frame, question: str = "Describe what you see in this image and the person's emotional state."):
        """
        Analyze a video frame (numpy array or OpenCV frame) using Gemini Vision
        
        Args:
            frame: Video frame (numpy array from OpenCV)
            question: Question about the frame
            
        Returns:
            AI description/analysis of the frame
        """
        if not GEMINI_AVAILABLE or self.model is None:
            return "⚠️ Gemini AI is not available."
        
        try:
            import PIL.Image
            import cv2
            
            # Convert BGR to RGB if it's an OpenCV frame
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to PIL Image
            img = PIL.Image.fromarray(frame_rgb)
            
            # Generate response
            response = self.model.generate_content([question, img])
            
            return response.text
            
        except Exception as e:
            error_msg = f"❌ Error analyzing frame: {str(e)}"
            print(error_msg)
            return error_msg
    
    def set_personality(self, personality: str):
        """
        Change AI personality mode
        
        Args:
            personality: One of 'empathetic', 'neutral', 'professional'
        """
        if personality in ["empathetic", "neutral", "professional"]:
            self.personality = personality
            print(f"🎭 Personality changed to: {personality}")
            
            # Restart chat to apply new personality
            if self.model:
                self.chat = self.model.start_chat(history=[])
        else:
            print(f"⚠️ Invalid personality: {personality}")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        if self.model:
            self.chat = self.model.start_chat(history=[])
        print("🗑️ Conversation history cleared")
    
    def save_conversation(self, filename: Optional[str] = None) -> str:
        """Save conversation history to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = SESSIONS_DIR / f"conversation_{timestamp}.json"
        
        # Ensure it's a Path object
        filename = Path(filename)
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "personality": self.personality,
            "total_messages": len(self.conversation_history),
            "conversation": self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Conversation saved to {filename}")
        return filename
    
    def get_conversation_summary(self) -> Dict:
        """Get summary statistics of the conversation"""
        if not self.conversation_history:
            return {
                "total_messages": 0,
                "personality": self.personality,
                "emotions_discussed": []
            }
        
        emotions = [msg.get("detected_emotion", "neutral") for msg in self.conversation_history]
        unique_emotions = list(set(emotions))
        
        return {
            "total_messages": len(self.conversation_history),
            "personality": self.personality,
            "emotions_discussed": unique_emotions,
            "most_common_emotion": max(set(emotions), key=emotions.count) if emotions else "neutral"
        }
    
    def get_suggestion(self, emotion: str) -> str:
        """
        Get AI suggestion based on detected emotion
        
        Args:
            emotion: Detected emotion
            
        Returns:
            Helpful suggestion or response
        """
        if not GEMINI_AVAILABLE or self.model is None:
            # Fallback suggestions
            suggestions = {
                "happy": "That's great! Keep up the positive energy! 😊",
                "sad": "I'm here if you want to talk about what's bothering you. 💙",
                "angry": "Take a deep breath. Would you like to talk about what's frustrating you?",
                "fear": "It's okay to feel anxious. Let's work through this together.",
                "surprise": "That was unexpected! Tell me more about it!",
                "disgust": "I understand that's unpleasant. How can I help?",
                "neutral": "How are you feeling today? I'm here to chat!"
            }
            return suggestions.get(emotion, "How can I help you today?")
        
        # Use Gemini for personalized suggestions
        try:
            prompt = (f"The user is feeling {emotion}. "
                     f"Provide a brief, {self.personality} response (1-2 sentences) "
                     f"acknowledging their emotion.")
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"❌ Error getting suggestion: {e}")
            return "How can I help you today?"
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model and available models
        
        Returns:
            Dictionary with model information
        """
        current_model = "None"
        if self.model:
            try:
                current_model = self.model.model_name
            except:
                current_model = "Unknown"
        
        return {
            "current_model": current_model,
            "available_models": self.available_models,
            "total_available": len(self.available_models),
            "api_configured": self.model is not None
        }
    
    def list_all_models(self, verbose: bool = False) -> List[Dict]:
        """
        List all available Gemini models with details
        
        Args:
            verbose: If True, include full details for each model
            
        Returns:
            List of model information dictionaries
        """
        if not GEMINI_AVAILABLE:
            return []
        
        try:
            models = genai.list_models()
            model_list = []
            
            for model in models:
                model_info = {
                    "name": model.name.replace('models/', ''),
                    "display_name": model.display_name,
                    "supports_generate": 'generateContent' in model.supported_generation_methods
                }
                
                if verbose:
                    model_info["supported_methods"] = list(model.supported_generation_methods)
                
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different Gemini model
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if not GEMINI_AVAILABLE:
            print("⚠️ Gemini not available")
            return False
        
        try:
            # Try to initialize the new model
            new_model = genai.GenerativeModel(model_name)
            self.model = new_model
            self.chat = self.model.start_chat(history=[])
            print(f"✅ Switched to model: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to switch model: {e}")
            return False
