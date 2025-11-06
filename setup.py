"""
Quick Setup Script for Emotion Detection System
Helps verify installation and dependencies
"""

import sys
import subprocess
import os

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required = [
        ("opencv-python", "cv2"),
        ("customtkinter", "customtkinter"),
        ("google-generativeai", "google.generativeai"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
    ]
    
    optional = [
        ("SpeechRecognition", "speech_recognition", "Voice input"),
        ("mediapipe", "mediapipe", "Gesture recognition"),
        ("pandas", "pandas", "CSV export"),
        ("matplotlib", "matplotlib", "Charts"),
    ]
    
    missing = []
    
    print("\nRequired packages:")
    for package, import_name in required:
        try:
            __import__(import_name)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    print("\nOptional packages:")
    for package, import_name, feature in optional:
        try:
            __import__(import_name)
            print(f"  ✅ {package} ({feature})")
        except ImportError:
            print(f"  ⚠️  {package} ({feature}) - Not installed")
    
    return len(missing) == 0

def check_files():
    """Check if required files exist"""
    print_header("Checking Files")
    
    files = [
        "config.json",
        "model/haarcascade_frontalface_default.xml",
        "src/main.py",
        "src/detector.py",
        "src/gemini_chatbot.py",
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_exist = False
    
    return all_exist

def check_api_key():
    """Check if Gemini API key is configured"""
    print_header("Checking API Configuration")
    
    try:
        import json
        with open("config.json", "r") as f:
            config = json.load(f)
        
        api_key = config.get("gemini_api_key", "")
        
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            print("  ⚠️  Gemini API key not configured")
            print("  Please edit config.json and add your API key")
            print("  Get one at: https://makersuite.google.com/app/apikey")
            return False
        else:
            print("  ✅ Gemini API key is configured")
            return True
    except Exception as e:
        print(f"  ❌ Error reading config: {e}")
        return False

def install_dependencies():
    """Install missing dependencies"""
    print_header("Installing Dependencies")
    print("\nRunning: pip install -r requirements.txt\n")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies")
        return False

def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("  🎯 Emotion Detection System - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        input("\nPress Enter to exit...")
        return
    
    # Check files
    files_ok = check_files()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Offer to install if missing
    if not deps_ok:
        print("\n⚠️  Some required packages are missing")
        response = input("\nWould you like to install them now? (y/n): ")
        
        if response.lower() == 'y':
            if install_dependencies():
                deps_ok = True
    
    # Check API key
    api_ok = check_api_key()
    
    # Final summary
    print_header("Setup Summary")
    
    print(f"  Files:        {'✅ OK' if files_ok else '❌ Issues'}")
    print(f"  Dependencies: {'✅ OK' if deps_ok else '❌ Issues'}")
    print(f"  API Key:      {'✅ OK' if api_ok else '⚠️  Not configured'}")
    
    if files_ok and deps_ok:
        print("\n✅ Setup complete! You can now run the application:")
        print("\n  Windows: run.bat")
        print("  Or: cd src && python main.py")
        
        if not api_ok:
            print("\n⚠️  Don't forget to configure your Gemini API key in config.json")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
