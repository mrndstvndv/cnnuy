# 🎯 Comprehensive Hybrid Emotion Detection System

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-brightgreen.svg)](https://opencv.org/)

Real-time emotion detection system combining **facial recognition**, **text analysis**, **voice-to-text chat**, **hand gestures**, **AI-powered vision**, and **text-to-speech responses**. Built with Python, TensorFlow, OpenCV, MediaPipe, Google Gemini AI, and ElevenLabs TTS.

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📋 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [📖 Usage](#-usage)
- [🎯 Core Features](#-core-features)
- [🎮 Complete Gesture Guide](#-complete-gesture-guide)
- [🎤 Voice Features](#-voice-features)
- [🔊 Text-to-Speech](#-text-to-speech)
- [🤖 AI Chatbot](#-ai-chatbot)
- [📊 Analytics](#-analytics)
- [🔧 Troubleshooting](#-troubleshooting)
- [🖥️ Screen Capture Guide](#️-screen-capture-guide)
- [🖼️ Image Upload Guide](#️-image-upload-guide)
- [📊 Performance Comparison by Input Mode](#-performance-comparison-by-input-mode)
- [🧪 Technical](#-technical)
- [📁 Structure](#-structure)

---

## ✨ Features

### 🎥 Multiple Input Sources (NEW!)
- **Webcam**: Real-time face detection from camera
- **Screen Capture**: Detect emotions from faces on your screen
- **Image Upload**: Analyze emotions in uploaded photos
- **Easy Switching**: Select input mode from dropdown menu

### 🎯 Emotion Detection
- **7 Emotions**: Angry🔴 Disgust🟢 Fear🟣 Happy🟡 Neutral⚪ Sad🟠 Surprise🔵
- **Real-Time**: CNN at 30 FPS with color-coded borders
- **Smoothing**: 1-20 frame averaging for stability (improved!)
- **Enhanced**: Works with glasses/varied lighting
- **Multi-Source**: Face, text, voice analysis
- **Glasses Warning**: Automatic reminder for best accuracy

### 🎤 Voice-to-Text Chat (NEW!)
- **Hands-Free**: Activate with ☝️ Pointing gesture
- **Auto-Transcribe**: Speech to text using Google
- **AI Responds**: Automatic chatbot reply
- **Natural Flow**: Speak → Transcribe → AI responds with voice!

### 🔊 Text-to-Speech (NEW!)
- **Dual System**: ElevenLabs (premium) + pyttsx3 (offline fallback)
- **Natural Voice**: High-quality AI voice responses
- **Auto-Playback**: AI speaks after responding
- **Toggle Control**: Easy on/off switch
- **API Testing**: Built-in connection and quota checker

### 🤖 AI Vision
- **Gemini AI**: Webcam scene analysis
- **Auto Mode**: Every 30 seconds
- **3 Personalities**: Empathetic/Neutral/Professional
- **Memory**: Full conversation context
- **Adaptive**: Responds to your emotions

### 🎮 Gestures (MediaPipe) - COMPLETELY REDESIGNED!
- **11 Gestures**: Intuitive, powerful controls
- **Voice Control**: ☝️ Start, 🤙 Stop
- **Video Control**: ✌️ Stop detection
- **Motion Control**: Hand up/down = brightness
- **Quick Actions**: Snapshot, save, clear, AI analyze
- **Smart Mapping**: Logical gesture meanings

### 📊 Analytics
- **Comparison**: Face vs text vs voice
- **Mismatch**: Authenticity scoring
- **Logging**: Timestamped sessions
- **Export**: JSON, CSV, Excel

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
python src/main.py

# Configure API keys in the app:
# Click ⚙️ Settings button → Enter API keys → Save
```

**Get API Keys**: 
- Gemini: https://makersuite.google.com/app/apikey (free, required for AI features)
- ElevenLabs: https://elevenlabs.io/ (optional, for premium TTS)

**OR manually edit config.json**:
```json
"gemini_api_key": "YOUR_GEMINI_KEY_HERE"
"elevenlabs_api_key": "YOUR_ELEVENLABS_KEY_HERE"  # Optional
``` 
- Gemini: https://makersuite.google.com/app/apikey (free)
- ElevenLabs: https://elevenlabs.io/ (optional, has offline fallback)

---

## 📋 Installation

### Python Version

| Version | Features | Voice | Gestures | Status |
|---------|----------|-------|----------|--------|
| 3.10    | ✅ Full  | ✅ Auto | ✅ Auto | ⭐ Excellent |
| 3.11    | ✅ Full  | ✅ Auto | ✅ Auto | ⭐ Excellent |
| 3.12    | ✅ Full  | ✅ Auto | ✅ Auto | ⭐⭐ **Best** |
| 3.13    | ✅ Full  | ⚠️ Manual | ❌ None | ⚠️ Limited |

### Dependencies

**Auto-Install:**
```bash
pip install -r requirements.txt
```

Includes: opencv-python, tensorflow, keras, google-generativeai, customtkinter, speechrecognition, matplotlib, pandas, numpy, pillow, openpyxl, pyttsx3, pygame

**Optional - PyAudio (Voice):**
```bash
# Windows (Python 3.10-3.12)
pip install pipwin
pipwin install pyaudio

# Python 3.13 - Manual
# Download .whl from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio-0.2.14-cp313-cp313-win_amd64.whl
```

**Optional - MediaPipe (Gestures):**
- Auto-installs on Python 3.9-3.12
- Not available for Python 3.13+

---

## ⚙️ Configuration

### Using Settings UI (Recommended ✨)

1. Launch the application: `python src/main.py`
2. Click **⚙️ Settings** button in the top control bar
3. Enter your API keys:
   - **Gemini API Key**: Required for AI features
   - **ElevenLabs API Key**: Optional for premium TTS
4. Adjust other settings:
   - Camera Index (0 for built-in, 1+ for external)
   - Screen Capture FPS (1-30, recommended: 5-10)
5. Click **💾 Save Settings**

**Benefits**:
- ✅ No need to manually edit JSON files
- ✅ Settings saved automatically
- ✅ Changes apply immediately
- ✅ API keys are hidden with dots (•••)
- ✅ Input validation built-in

### Manual Configuration (Alternative)

Edit `config.json` directly:

```json
{
  "gemini_api_key": "YOUR_KEY_HERE",
  "elevenlabs_api_key": "YOUR_ELEVENLABS_KEY",
  "camera_index": 0,
  "confidence_threshold": 0.5,
  "smoothing_frames": 5,
  "theme": "dark",
  "enable_voice": true,
  "enable_gestures": true,
  "ai_personality": "empathetic",
  "auto_ai_vision": false
}
```

---

## 📖 Usage

### Basic Operations

**Choose Input Mode**
1. Select from dropdown menu: **Webcam** / **Screen** / **Image**
   - **Webcam**: Standard camera-based detection
   - **Screen**: Captures your entire screen to find faces (great for video calls!)
   - **Image**: Upload and analyze static photos

**Start Detection (Webcam/Screen)**
1. Select input mode
2. Click ▶️ Start Detection
3. For Webcam: Face the camera
4. For Screen: Ensure faces are visible on screen
5. Emotions display with colored borders

**Upload Image**
1. Select "Image" mode OR click 🖼️ Upload Image button
2. Choose image file (PNG, JPG, BMP, GIF)
3. View detected emotions and AI analysis
4. Emotion bars update with detected feelings

**Text Analysis**
1. Type in text box
2. Click Analyze Text
3. View emotion + sentiment score

**Voice Input**
1. Click 🎤 Voice Input
2. Speak clearly
3. View transcription + emotion

**AI Chat**
1. Type message in chat panel
2. Press Enter
3. AI responds based on your emotion

### Advanced

**Smoothing**: Adjust slider (1-10 frames)

**Brightness**: Hand up/down OR slider

**Screen Capture Settings**: 
- Screen capture runs at 5 FPS (configurable) to reduce CPU load
- Automatically detects faces on entire screen
- Useful for analyzing emotions during video calls or watching videos

**AI Vision**:
- Manual: Click 🤖 AI Vision button
- Auto: Toggle "Auto AI Vision (30s)"
- Gesture: Show OK sign 👌

**Mismatch Detection**: Enable checkbox → Compare face/text/voice

---

## 🎯 Core Features

### Facial Emotion Detection

**Tech**: CNN (Convolutional Neural Network)

**Colors**:
- Angry🔴 #FF0000 (furrowed brows)
- Disgust🟢 #008000 (wrinkled nose)
- Fear🟣 #800080 (wide eyes)
- Happy🟡 #FFFF00 (smile, crinkled eyes)
- Neutral⚪ #808080 (relaxed)
- Sad🟠 #FF8000 (downturned mouth)
- Surprise🔵 #00FFFF (open mouth)

**Features**:
- 30 FPS detection
- Multi-face support
- Confidence %
- Histogram equalization for lighting
- Scale factor 1.05 for thorough scanning

### Text Emotion

**Method**: Keyword matching

**Keywords**:
- Happy: love, joy, excited, wonderful
- Sad: depressed, unhappy, miserable
- Angry: furious, mad, hate
- Fear: scared, terrified, anxious

### Voice Emotion

**Tech**: SpeechRecognition + Google API

**Process**:
1. Mic captures audio
2. Speech-to-text
3. Keyword analysis
4. Result + transcription

**Tips**: Speak clearly, minimize noise, 6-12" from mic

---

## 🎮 Complete Gesture Guide

### 💡 Important Tips for Best Results
- 👓 **Remove glasses** for better emotion detection accuracy
- 💡 Ensure good lighting on your face
- 📏 Keep face centered and at comfortable distance from camera
- 👋 Hold gestures steady for 5 frames (about 0.5 seconds) to activate
- 🎯 For motion control, move hand smoothly and deliberately
- 🤙 Use Call Me gesture (thumb+pinky) to stop voice or AI speaking

### 📸 Snapshot & Recording
| Gesture | Hand Position | Action | Icon |
|---------|--------------|--------|------|
| **High Five** | All 5 fingers extended | Take snapshot | 🖐️ |
| **Thumbs Up** | Thumb only up | Save snapshot to exports | 👍 |

### 🎤 Voice & Audio Controls  
| Gesture | Hand Position | Action | Icon |
|---------|--------------|--------|------|
| **Pointing** | Index finger only | **START voice chat** | ☝️ |
| **Call Me** | Thumb + Pinky (phone gesture) | **STOP voice/speaking** | 🤙 |
| **Four Fingers** | 4 fingers up, no thumb | Toggle voice responses on/off | 🖖 |

### 🎬 Video & Detection Controls
| Gesture | Hand Position | Action | Icon |
|---------|--------------|--------|------|
| **Peace Sign** | Index + Middle fingers | **STOP video detection** | ✌️ |

### 🤖 AI & Analysis
| Gesture | Hand Position | Action | Icon |
|---------|--------------|--------|------|
| **OK Sign** | Thumb+Index circle, others up | AI vision analyze frame | � |
| **Three Fingers** | Index + Middle + Ring | Toggle auto-analysis | 🤟 |

### 💬 Chat & Interface
| Gesture | Hand Position | Action | Icon |
|---------|--------------|--------|------|
| **Thumbs Down** | Thumb pointing down | Clear AI chat history | 👎 |
| **Pinky Only** | Just pinky extended | Show help dialog | 🤙 |

### 🔆 Brightness Control (Motion-based)
| Gesture | Motion | Action | Display |
|---------|--------|--------|---------|
| **Any Gesture** | Move hand UP ⬆️ | Brightness +10% | ⬆️ |
| **Any Gesture** | Move hand DOWN ⬇️ | Brightness -10% | ⬇️ |
| **Fist** | Move up/down | Best for motion control | ✊ |

### 🎯 Usage Flow Examples

**Example 1: Voice Chat**
```
1. Show ☝️ POINTING → Voice recording starts
2. Speak your message → "What's the weather?"
3. AI transcribes and responds with text + voice
4. If AI talking too long → Show 🤙 CALL ME → Voice stops
```

**Example 2: Stop Everything**
```
1. ✌️ PEACE SIGN → Stops video detection
2. 🤙 CALL ME → Stops any speaking
3. Done!
```

**Example 3: Quick Snapshot**
```
1. 🖐️ HIGH FIVE → Takes snapshot
2. 👍 THUMBS UP → Saves to exports folder
```

**Example 4: Brightness Control**
```
1. Make ✊ FIST
2. Move hand UP → Brightness increases
3. Move hand DOWN → Brightness decreases
```

### 🔧 Gesture Stability Settings
- **Hold Time**: 0.5 seconds (5 frames at 30 FPS)
- **Cooldown**: 2 seconds between actions
- **Motion Threshold**: 0.08 (smooth, deliberate movements)
- **Stability**: 4 out of 5 frames must show same gesture (80% consistency)

### 💡 Gesture Detection Improvements
- ✅ **No more flickering** - Multi-level stabilization system
- ✅ **Motion control works** - Increased threshold to 0.08
- ✅ **OK sign detected** - Moved to first priority in detection
- ✅ **Clear meanings** - Logical gesture-to-action mapping
- ✅ **Better feedback** - Live display shows gesture → action

---

## 🎤 Voice Features

### Voice-to-Text Chat

**How It Works:**
1. Show 🤙 Call Me gesture OR click 🎤 Voice Input button
2. System starts listening (up to 10 seconds)
3. Speak your message naturally
4. Voice automatically transcribed to text using Google Speech Recognition
5. AI receives message and generates response
6. AI speaks response out loud (if TTS enabled)

**Visual Feedback:**
- 🎤 "Voice recording started... Speak now!" (blue)
- 📝 "You said: [your message]" (user)
- 🤖 "AI: [response]" (AI text)
- 🔊 "AI Speaking..." (during playback)

**Requirements:**
- **Microphone**: Working and properly configured
- **Internet**: Required for Google Speech Recognition
- **PyAudio**: Installed (auto-installs on Python 3.10-3.12)

**Error Handling:**
- ❌ "No speech detected" - Speak louder or closer to mic
- ❌ "Could not understand" - Speak more clearly, reduce background noise
- ⏱️ Timeout after 10 seconds

### Features
- ✅ **Hands-free operation** - Gesture activates voice
- ✅ **Auto-transcription** - Speech to text
- ✅ **AI integration** - Sends to chatbot automatically
- ✅ **Context-aware** - AI knows your current emotion
- ✅ **Voice responses** - AI speaks back (if enabled)

---

## 🔊 Text-to-Speech

### Dual TTS System

The app uses a **two-tier text-to-speech system** for maximum reliability:

#### 🌟 Primary: ElevenLabs (High Quality)
- **Voice**: Rachel (friendly female, natural AI voice)
- **Quality**: Professional-grade synthesis
- **Requires**: API key + internet connection
- **Status**: Optional (automatically falls back if unavailable)

#### 🎤 Fallback: pyttsx3 (Offline)
- **Voice**: Windows SAPI voices (built-in)
- **Quality**: System voices (good, but robotic)
- **Requires**: Only pyttsx3 library (already installed)
- **Status**: ✅ **ALWAYS AVAILABLE** - Works offline!

### Testing ElevenLabs API

**Button**: Click "🔊 Test TTS" in the top control bar

**What it shows:**
```
✅ ElevenLabs API Connected!
📊 Usage: 1,247 / 10,000 characters
📈 Remaining: 8,753 characters

🔊 TTS Test Successful! Playing audio...
```

**Or if there's an issue:**
```
❌ TTS Permission Error
The API key you used is missing the permission text_to_speech...

✅ API key is valid but lacks text-to-speech permission.
🔄 Using offline voice (pyttsx3) instead.
```

**Possible Results:**
- ✅ **Connected**: Shows character usage and remaining quota
- ❌ **No Permission**: API key lacks text_to_speech permission
- ❌ **Quota Exceeded**: Monthly limit reached
- ❌ **Invalid Key**: API key is wrong
- ⏱️ **Timeout**: No internet connection

### Toggle Voice Responses

**Switch**: "🔊 Voice Responses" in AI Chat panel

**When ON:**
- AI speaks responses using available TTS
- Shows: "🔊 Voice responses enabled (using offline voice)"

**When OFF:**
- AI only shows text responses
- No audio playback

### Voice Settings

**ElevenLabs Configuration** (in code):
```python
voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
voice_settings = {
    "stability": 0.5,        # 0-1: Lower = more expressive
    "similarity_boost": 0.75  # 0-1: Higher = better quality
}
```

**pyttsx3 Configuration** (in code):
```python
rate = 150  # Words per minute
volume = 0.9  # 0.0 to 1.0
```

### API Usage & Limits

**ElevenLabs Free Tier:**
- 10,000 characters/month
- Average AI response: ~100-200 characters
- Estimated: ~50-100 voice responses per month
- Monitor at: https://elevenlabs.io/

**Auto-Fallback:**
System automatically uses offline voice if:
- ElevenLabs unavailable
- No API key configured
- Quota exceeded
- Internet connection lost
- API error occurs

---

## 🤖 AI Chatbot

### Models (Auto-Selected)

1. **gemini-2.5-flash** ⭐ (Best balance)
2. gemini-flash-latest (Latest stable)
3. gemini-2.5-pro (Higher quality)
4. gemini-pro-latest
5. gemini-2.0-flash

### Vision Analysis

**Manual**: 🤖 AI Vision button → Frame analysis

**Auto**: Toggle "Auto AI Vision (30s)" → Every 30s

**Upload**: 🖼️ Upload → Image analysis

**Sees**: Face, eyes, body language, environment, emotion, appearance, scene

### Personalities

**Empathetic** (Default):
- Warm, understanding
- Emotional support
> "I see you're happy! What's bringing you joy?"

**Neutral**:
- Balanced, objective
- Fact-focused
> "Emotion: Happy at 85% confidence"

**Professional**:
- Concise, solution-oriented
- Business tone
> "Happy detected. How may I assist?"

### Features

- 💾 Save conversation
- 🔄 Clear chat
- 📊 Summary (messages, emotions)
- 🧠 Context memory
- 🎭 Emotion aware

**API Limits** (Free): 15 req/min, 1M tokens/min, 1.5K req/day

---

## 📊 Analytics

### Session Logging

**Auto-Captures**:
- ⏰ Timestamps
- 🎭 Emotion + confidence
- 📱 Source (face/text/voice)
- 💬 Chat history

**Structure**:
```json
{
  "session_id": "20251030_143052",
  "total_detections": 1247,
  "emotions_summary": {"happy": 450, ...},
  "data": [{"timestamp": "...", "emotion": "happy", ...}]
}
```

### Exports

1. **JSON**: Full data → `data/exports/session_TIME.json`
2. **CSV**: Spreadsheet → `data/exports/session_TIME.csv`
3. **Excel**: Multi-sheet → `data/exports/session_TIME.xlsx`
4. **Chat**: Text → `data/exports/conversation_TIME.txt`

### Visuals

- **Bars**: Color-coded emotion probabilities
- **Pie**: Emotion distribution
- **Timeline**: Changes over time
- **Comparison**: Face vs text vs voice

### Mismatch

**Detects**: Emotional incongruence

**Types**:
- Face vs Text
- Face vs Voice
- Text vs Voice

**Metrics**:
- Authenticity 0-100%
- Mismatch alert
- Consistency tracking

**Example**:
```
Face: Happy (85%)
Text: Sad (78%)
⚠️ MISMATCH - Authenticity: 42%
```

---

## 🔧 Troubleshooting

### Voice & TTS Issues

#### "No module 'pyaudio'"
- **Python 3.10-3.12**: `pipwin install pyaudio`
- **Python 3.13**: Download .whl from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
- **Quick Fix**: Disable voice in config: `"enable_voice": false`

#### Voice Input Not Working
- ✅ **Check microphone**: Speak test in Windows Sound settings
- ✅ **Grant permissions**: Windows Settings > Privacy > Microphone
- ✅ **Internet required**: Google Speech Recognition needs connection
- ✅ **Speak clearly**: Reduce background noise, speak 6-12" from mic
- ✅ **Check logs**: "No speech detected" = speak louder, "Could not understand" = speak clearer

#### ElevenLabs TTS Issues
**❌ "TTS Permission Error"**
```
The API key you used is missing the permission text_to_speech...
```
- **Cause**: API key lacks text_to_speech permission
- **Solution**: App auto-falls back to offline voice (pyttsx3)
- **Optional**: Get new API key with correct permissions at https://elevenlabs.io/

**❌ "Quota Exceeded"**
```
Free tier quota exceeded (10,000 chars/month)
```
- **Solution**: Wait for monthly reset OR upgrade plan
- **Fallback**: pyttsx3 still works offline

**❌ "Invalid API Key"**
- **Check**: config.json → "elevenlabs_api_key" field
- **Fix**: Remove extra spaces/quotes, verify key at https://elevenlabs.io/

**⏱️ "Request Timeout"**
- **Cause**: No internet connection
- **Fallback**: Automatically uses pyttsx3 offline voice

#### pyttsx3 Issues
**❌ "No voices available"**
- **Windows**: Reinstall "Microsoft Speech Platform"
- **Fix**: Install Windows SAPI voices (should be built-in)

**❌ "Audio playback error"**
- **pygame required**: `pip install pygame`
- **Check volume**: System audio not muted

### Gesture Issues

#### "MediaPipe not available"
- **Python 3.13+**: MediaPipe not yet supported
- **Solution**: Use Python 3.10-3.12 for gesture features
- **Quick Fix**: Disable in config: `"enable_gestures": false`

#### Gestures Keep Flickering/Changing
- **Fix Applied**: Stabilization system requires 4/5 frames consistency
- **Tips**:
  - Hold gesture steady for 0.5 seconds (5 frames)
  - Keep hand stable in camera view
  - Ensure good lighting
  - Avoid rapid hand movements
  - Palm should face camera

#### Motion Control (Up/Down) Not Working
- **Fix Applied**: Motion threshold increased to 0.08
- **Tips**:
  - Move hand **smoothly and deliberately**
  - Use ✊ FIST for best motion control
  - Motion requires 0.08 vertical movement (~8% of screen)
  - Move at steady pace (not too fast/slow)
  - System tracks 15 frames for stability

#### OK Sign 👌 Not Detected
- **Fix Applied**: OK sign moved to first priority detection
- **Tips**:
  - Make clear circle with thumb + index finger
  - Extend other 3 fingers
  - Hold steady for 0.5 seconds
  - Face palm toward camera

#### Glasses Detection Warning
- **"Please remove glasses for better accuracy"**
- **Cause**: Glasses can interfere with emotion detection
- **Solution**: Remove glasses or ignore warning (still works, just less accurate)
- **Color**: Orange message in chat log

### General Issues

#### "API key not valid" (Gemini AI)
- Get key: https://makersuite.google.com/app/apikey
- Edit `config.json` → "gemini_api_key"
- Remove spaces/quotes from key

#### "Model not found"
- Check `model/best_emotion_model.keras` exists
- System uses fallback if missing
- Re-run setup if needed

#### Camera won't start
- Check USB connection
- Try different camera: `"camera_index": 1` or `2` in config.json
- Close other apps using camera (Zoom, Teams, Skype)
- Grant permissions: Windows Settings > Privacy > Camera
- Restart app

#### Low FPS / Laggy
- Reduce smoothing to 1-3 frames
- Disable gestures: `"enable_gestures": false`
- Disable voice: `"enable_voice": false`
- Close other resource-heavy apps
- Use lower resolution camera

#### Rapid emotion changes / Jittery detection
- Increase smoothing to 5-10 frames
- Improve lighting (face should be well-lit, even lighting)
- Keep face centered and still
- Remove glasses

#### AI not responding
- **Check internet connection**
- **Verify API key** in config.json
- **Check quota**: https://aistudio.google.com/app/apikey
- **Wait if rate limited** (too many requests)
- **Try again** in a few seconds

### Screen Capture Issues

#### "Screen capture not available"
**Error**: "Screen capture not available. Install 'mss' package."

**Solution**:
```bash
pip install mss
```

#### Permission Denied (macOS)
**Error**: Screen recording permission denied

**Solution**:
1. Quit the application
2. Go to **System Preferences** → **Security & Privacy** → **Privacy** → **Screen Recording**
3. Enable permission for **Terminal** or **Python**
4. Restart the application

#### Black Screen / No Faces Detected
**Possible causes**:
- Faces are too small on screen (zoom in the video call)
- Poor lighting in the video feed
- Faces partially obscured

**Solutions**:
- Increase the size of video windows
- Ensure faces are clearly visible
- Adjust screen brightness/contrast

#### High CPU Usage in Screen Mode
**Solution**:
- Screen capture is resource-intensive
- App limits capture to 5 FPS by default
- Close unnecessary applications
- Consider using webcam mode for extended sessions
- Adjust FPS in code: `self.screen_fps = 3` (lower = less CPU)

---

## 🖥️ Screen Capture Guide

### Overview

Screen capture mode detects faces and emotions from **anywhere on your screen** - perfect for video calls, watching videos, or analyzing content!

### Setup

**Install required library**:
```bash
pip install mss
# Or install all dependencies
pip install -r requirements.txt
```

**Platform-Specific Permissions**:

| Platform | Permissions Required |
|----------|---------------------|
| **Windows 10/11** | ✅ None (works out of the box) |
| **macOS** | ⚠️ Screen Recording permission required |
| **Linux** | ✅ Usually none (X11 dev files if needed) |

### How to Use Screen Capture

1. Select **"screen"** from the **Input:** dropdown menu at the top
2. Click **▶️ Start** button
3. The app will:
   - Capture your primary monitor at 5 FPS
   - Detect any faces visible on screen
   - Analyze emotions in real-time
   - Draw colored detection boxes around faces

### Use Cases

✅ **Video Calls**: Analyze emotions during Zoom/Teams/Meet  
✅ **Content Analysis**: Study reactions while watching videos  
✅ **Multiple People**: Detect emotions from multiple faces  
✅ **Presentations**: Monitor audience reactions  
✅ **Remote Work**: Understand team emotions in virtual meetings  

### Performance

- **FPS**: 5 (vs 30 for webcam) - optimized for low CPU usage
- **CPU**: Low-Medium (~30-40%)
- **Best For**: Video calls, content analysis, multi-person detection

### Configuration

Adjust screen capture FPS in `src/main.py` (line ~75):

```python
self.screen_fps = 5  # Adjust this value (1-30)
```

**Recommendations**:
- **1-3 FPS**: Minimal CPU, slower updates
- **5-10 FPS**: Balanced (recommended)
- **15-30 FPS**: Smoother, high CPU usage

Select specific monitor (multi-monitor setups):

```python
# In update_frame() method
monitor = self.screen_capturer.monitors[1]  # Primary
monitor = self.screen_capturer.monitors[2]  # Secondary
```

### Privacy & Security

🔒 **Your privacy matters**:
- Screen capture only runs when explicitly started
- No data sent to external servers (except optional AI)
- All processing happens locally on your device
- Stop anytime with ⏹️ Stop button

---

## 🖼️ Image Upload Guide

### Overview

Upload and analyze static images with **full emotion detection** plus AI analysis!

### Features

✅ Upload photos (PNG, JPG, JPEG, BMP, GIF)  
✅ Automatic face detection  
✅ Full emotion analysis with color-coded results  
✅ Emotion bars show all 7 emotions  
✅ Dual analysis: Emotion detection + Gemini AI  
✅ Results in chat with percentages  

### How to Use

**Method 1**: Via Input Mode
1. Select **"image"** from Input dropdown
2. Application prompts for file selection
3. Choose your image
4. View instant analysis

**Method 2**: Direct Upload (works in any mode)
1. Click **🖼️ Upload Image** button
2. Select image file
3. View results immediately

### What You'll See

- **Annotated Image**: Face boxes with emotion labels
- **Emotion Bars**: Updated percentages for all 7 emotions
- **Dominant Emotion**: Highlighted with confidence %
- **AI Description**: Gemini AI analyzes the image
- **Chat Summary**: "Detected X face(s). Dominant emotion: [emotion] (XX%)"

### Best Results

💡 **Tips**:
- Clear, well-lit faces
- Front-facing photos work best
- Higher resolution = better detection
- Remove glasses for accuracy
- Neutral background helps

---

## 📊 Performance Comparison by Input Mode

| Mode | FPS | CPU Usage | Memory | Best For |
|------|-----|-----------|--------|----------|
| **Webcam** | 30 | ~25% | ~100MB | Real-time personal detection |
| **Screen** | 5 | ~35% | ~120MB | Video calls, content analysis |
| **Image** | N/A | ~5% | ~80MB | Photo analysis, testing |

**Component Breakdown**:
- Face Detection: ~25% CPU
- Emotion Model: ~10% CPU
- Gestures: ~15% CPU (if enabled)
- AI Vision: ~5% CPU
- GUI: ~5% CPU

---

## 🧪 Technical

### Emotion Model

- **Architecture**: CNN
- **Input**: 48x48 grayscale
- **Output**: 7 classes
- **Dataset**: FER-2013
- **Accuracy**: ~85%

**Preprocessing**: Grayscale → Histogram eq → Crop → Normalize

### Face Detection

- **Method**: Haar Cascade (OpenCV)
- **Params**: scaleFactor=1.05, minNeighbors=3, minSize=60x60
- **Features**: Multi-face, 30 FPS, lighting normalization

### Gestures

- **Tech**: MediaPipe Hands
- **Landmarks**: 21 per hand
- **Coordinates**: 3D (x,y,z)
- **Accuracy**: 85-95%

**Algorithm**:
```python
fingers_up = sum([thumb_up, index_up, ...])
if fingers_up == 5: gesture = "high_five"
elif fingers_up == 0: gesture = "fist"
```

---

## 📁 Structure

```
emotion-detection-system/
│
├── config.json             # Configuration
├── requirements.txt        # Dependencies
├── README.md              # This file
│
├── model/
│   └── best_emotion_model.keras
│
├── src/
│   ├── main.py            # Entry point
│   ├── detector.py        # Emotion detection
│   ├── gemini_chatbot.py  # AI integration
│   ├── test_*.py          # Tests
│   └── __pycache__/
│
├── data/
│   ├── sessions/          # Auto-saved JSON
│   └── exports/           # User exports
│
└── cascades/
    └── haarcascade_frontalface_default.xml
```

---

## 🔒 Privacy

**Collected**: Emotions, timestamps, chat, metadata

**NOT Collected**: Raw video/audio, PII, external uploads

**Storage**: Local only (`data/` folder)

**API**: Only explicit sends to Gemini (Google privacy policy)

**Security**:
- Don't share config.json
- Don't commit API keys
- Rotate keys periodically
- Camera only active when started

---

## 🤝 Contributing

**Issues**: Include Python version, OS, errors

**Features**: Open issue with [Feature Request]

**Code**:
1. Fork repo
2. Create branch: `git checkout -b feature/name`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin feature/name`
5. Open Pull Request

---

## 📝 License

MIT License - Copyright (c) 2025

---

## 🙏 Acknowledgments

**Tech**: TensorFlow, OpenCV, MediaPipe, Gemini AI, CustomTkinter

**Data**: FER-2013, CK+

---

**Built with ❤️ for emotion recognition**

*Last Updated: November 6, 2025*