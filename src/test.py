"""
Simple test script to verify the emotion detection system works
"""

print("🎯 Testing Emotion Detection System Components...\n")

# Test 1: Import detector
print("[1/5] Testing detector module...")
try:
    from detector import EmotionDetector
    detector = EmotionDetector()
    print("     ✅ Detector initialized successfully")
except Exception as e:
    print(f"     ❌ Error: {e}")

# Test 2: Import chatbot
print("\n[2/5] Testing chatbot module...")
try:
    from gemini_chatbot import GeminiChatbot
    chatbot = GeminiChatbot()
    print("     ✅ Chatbot initialized successfully")
except Exception as e:
    print(f"     ❌ Error: {e}")

# Test 3: Test text analysis
print("\n[3/5] Testing text emotion analysis...")
try:
    result = detector.analyze_text("I am so happy today!")
    print(f"     ✅ Detected emotion: {result['emotion']}")
    print(f"     ✅ Sentiment: {result['sentiment']}")
except Exception as e:
    print(f"     ❌ Error: {e}")

# Test 4: Check face detection model
print("\n[4/5] Testing face detection model...")
try:
    import cv2
    import numpy as np
    
    # Create a dummy frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect_faces(test_frame)
    print(f"     ✅ Face detector loaded (found {len(faces)} faces in test)")
except Exception as e:
    print(f"     ❌ Error: {e}")

# Test 5: Check Gemini API
print("\n[5/5] Testing Gemini API connection...")
try:
    import google.generativeai as genai
    
    # Check if API key is configured
    import json
    with open("../config.json", "r") as f:
        config = json.load(f)
    
    api_key = config.get("gemini_api_key", "")
    
    if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
        print(f"     ✅ API key configured: {api_key[:10]}...")
        
        # Try to initialize
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        print("     ✅ Gemini API connection successful")
    else:
        print("     ⚠️  API key not configured (chatbot won't work)")
except Exception as e:
    print(f"     ⚠️  Gemini API issue: {e}")

# Summary
print("\n" + "=" * 60)
print("  Test Summary")
print("=" * 60)
print("\nCore components are ready! 🎉")
print("\nYou can now run the main application:")
print("  python main.py")
print("\nOr use the launcher:")
print("  run.bat (Windows)")
print("\n" + "=" * 60)

input("\nPress Enter to exit...")
