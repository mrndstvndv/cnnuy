"""
Test script for the enhanced Gemini Chatbot with model detection
"""

from gemini_chatbot import GeminiChatbot

def main():
    print("=" * 70)
    print("  GEMINI CHATBOT - MODEL DETECTION TEST")
    print("=" * 70)
    
    # Initialize chatbot
    print("\n1️⃣  Initializing chatbot...")
    bot = GeminiChatbot()
    
    # Get model info
    print("\n2️⃣  Model Information:")
    info = bot.get_model_info()
    print(f"   • Current Model: {info['current_model']}")
    print(f"   • API Configured: {info['api_configured']}")
    print(f"   • Available Models: {info['total_available']}")
    
    # Show first 5 available models
    print("\n3️⃣  Top 5 Available Models:")
    for i, model in enumerate(info['available_models'][:5], 1):
        print(f"   {i}. {model}")
    
    # Test conversation
    print("\n4️⃣  Testing conversation...")
    response = bot.send_message("Hello! How are you?", "happy")
    print(f"   User: Hello! How are you? (emotion: happy)")
    print(f"   AI: {response[:100]}...")
    
    # Get conversation summary
    print("\n5️⃣  Conversation Summary:")
    summary = bot.get_conversation_summary()
    print(f"   • Total Messages: {summary['total_messages']}")
    print(f"   • Personality: {summary['personality']}")
    print(f"   • Emotions: {summary['emotions_discussed']}")
    
    # Test emotion suggestion
    print("\n6️⃣  Testing emotion-based suggestion:")
    suggestion = bot.get_suggestion("sad")
    print(f"   Emotion: sad")
    print(f"   Suggestion: {suggestion}")
    
    # List all models (brief)
    print("\n7️⃣  All Available Models (Brief):")
    all_models = bot.list_all_models(verbose=False)
    text_models = [m for m in all_models if m['supports_generate'] and 'gemini' in m['name'].lower()]
    print(f"   Found {len(text_models)} text-capable Gemini models")
    for model in text_models[:10]:
        emoji = "✅" if model['supports_generate'] else "❌"
        print(f"   {emoji} {model['name']}")
    
    print("\n" + "=" * 70)
    print("  ✅ All tests completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
