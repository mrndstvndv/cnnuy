"""
Test Gemini Vision with a sample frame
"""

import cv2
import numpy as np
from gemini_chatbot import GeminiChatbot

def create_test_frame():
    """Create a simple test frame"""
    # Create a blank image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some text
    cv2.putText(frame, "TEST FRAME", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    return frame

def main():
    print("=" * 70)
    print("  GEMINI VISION - FRAME ANALYSIS TEST")
    print("=" * 70)
    
    # Initialize chatbot
    print("\n1️⃣  Initializing chatbot...")
    bot = GeminiChatbot()
    
    # Create test frame
    print("\n2️⃣  Creating test frame...")
    frame = create_test_frame()
    
    # Analyze frame
    print("\n3️⃣  Analyzing frame with Gemini Vision...")
    response = bot.analyze_frame(frame, "What text do you see in this image?")
    
    print(f"\n📹 AI Response:")
    print(f"   {response}")
    
    # Test with webcam (if available)
    print("\n4️⃣  Testing with webcam (if available)...")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, webcam_frame = cap.read()
        if ret:
            print("   ✅ Webcam frame captured")
            response = bot.analyze_frame(
                webcam_frame,
                "Describe what you see. Is there a person? What's their expression?"
            )
            print(f"\n📹 Webcam Analysis:")
            print(f"   {response}")
        else:
            print("   ⚠️ Could not read from webcam")
        cap.release()
    else:
        print("   ⚠️ Webcam not available")
    
    print("\n" + "=" * 70)
    print("  ✅ Test completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
