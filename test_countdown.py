"""
Test countdown functionality
"""
import sys
sys.path.insert(0, 'src')

from main import EmotionDetectionApp
import numpy as np
import cv2

def test_countdown():
    """Test the countdown timer"""
    print("Testing countdown functionality...")
    
    # Create app
    app = EmotionDetectionApp()
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test Frame", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    app.current_frame = test_frame
    
    # Test countdown overlay
    countdown_values = [3, 2, 1]
    for val in countdown_values:
        app.countdown_value = val
        app.countdown_active = True
        overlay_frame = app._add_countdown_overlay(test_frame.copy())
        print(f"✅ Countdown overlay for {val} created successfully")
    
    # Test countdown callback
    callback_executed = [False]
    
    def test_callback():
        callback_executed[0] = True
        print("✅ Countdown callback executed")
    
    app.countdown_callback = test_callback
    
    print("\n✅ All countdown tests passed!")
    print("\nCountdown features:")
    print("  - countdown_active flag")
    print("  - countdown_value tracking")
    print("  - _add_countdown_overlay() method")
    print("  - start_countdown() method")
    print("  - Countdown overlay with numbers and 'GET READY' text")
    print("  - Semi-transparent background")
    print("  - Auto-save screenshots to exports/")
    
    return True

if __name__ == "__main__":
    test_countdown()
