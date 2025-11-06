"""
Test gesture detection with MediaPipe
"""

import cv2
import numpy as np
from detector import EmotionDetector

def main():
    print("=" * 70)
    print("  HAND GESTURE DETECTION TEST")
    print("=" * 70)
    
    # Initialize detector
    print("\n1️⃣  Initializing detector...")
    detector = EmotionDetector()
    
    # Check MediaPipe availability
    if detector.hands is None:
        print("\n❌ MediaPipe not available!")
        print("   Install with: pip install mediapipe")
        return
    
    print("✅ MediaPipe hand tracking ready!")
    
    # Test with webcam
    print("\n2️⃣  Opening webcam for gesture detection...")
    print("   (Press 'q' to quit)")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    gesture_counts = {}
    frame_count = 0
    
    print("\n3️⃣  Detecting gestures...")
    print("\n   Recognized Gestures:")
    print("   - Open Palm (5 fingers) 🖐️")
    print("   - Fist (0 fingers) ✊")
    print("   - Peace (2 fingers) ✌️")
    print("   - Pointing (1 finger) 👉")
    print("   - Thumbs Up 👍")
    print("   - Call Me (thumb + pinky) 🤙")
    print("\n   Show gestures to the camera!")
    print("   " + "-" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect gestures
        frame, gesture, gesture_data = detector.detect_gestures(frame)
        frame_count += 1
        
        # Count gestures
        if gesture != "none" and gesture != "unknown":
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Display info on frame
        cv2.putText(
            frame, f"Frames: {frame_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Get emotion from gesture
        if gesture != "none":
            emotion = detector.get_gesture_emotion_mapping(gesture)
            cv2.putText(
                frame, f"Emotion: {emotion}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
        
        # Show frame
        cv2.imshow("Gesture Detection Test", frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Show statistics
    print("\n" + "=" * 70)
    print("  📊 DETECTION STATISTICS")
    print("=" * 70)
    print(f"\n   Total Frames: {frame_count}")
    print(f"\n   Gestures Detected:")
    
    if gesture_counts:
        for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / frame_count) * 100
            emoji_map = {
                "open_palm": "🖐️", "fist": "✊", "peace": "✌️",
                "pointing": "👉", "thumbs_up": "👍", "call_me": "🤙"
            }
            emoji = emoji_map.get(gesture, "❓")
            gesture_name = gesture.replace("_", " ").title()
            print(f"   {emoji} {gesture_name:15} - {count:4} times ({percentage:5.1f}%)")
    else:
        print("   No gestures detected")
    
    print("\n" + "=" * 70)
    print("  ✅ Test completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
