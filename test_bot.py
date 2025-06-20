#!/usr/bin/env python3
"""
Test script for HistoryBot functionality
This script tests the core features without running the Discord bot
"""

import json
import os
import sys
from datetime import datetime

# Add the current directory to the path so we can import from bot.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock Discord message for testing
class MockMessage:
    def __init__(self, content, author_name, author_id="123", channel_name="test", guild_name="test_server"):
        self.content = content
        self.author = MockAuthor(author_name, author_id)
        self.channel = MockChannel(channel_name)
        self.guild = MockGuild(guild_name) if guild_name else None
        self.id = "msg_" + str(hash(content))
        self.created_at = datetime.now()

class MockAuthor:
    def __init__(self, display_name, id):
        self.display_name = display_name
        self.id = id
        self.bot = False

class MockChannel:
    def __init__(self, name):
        self.name = name
        self.id = "channel_123"

class MockGuild:
    def __init__(self, name):
        self.name = name
        self.id = "guild_123"

def test_embedding_functionality():
    """Test the embedding and similarity functionality"""
    print("🧪 Testing embedding functionality...")
    
    try:
        # Import the bot class
        from bot import HistoryBot
        
        # Create a test instance
        bot = HistoryBot()
        
        # Test embedding generation
        test_text = "Hello, this is a test message"
        embedding = bot.get_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print("✅ Embedding generation works")
        else:
            print("❌ Embedding generation failed")
            return False
        
        # Test cosine similarity
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = bot.cosine_similarity(vec1, vec2)
        
        if similarity == 1.0:
            print("✅ Cosine similarity calculation works")
        else:
            print("❌ Cosine similarity calculation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing embedding functionality: {e}")
        return False

def test_message_storage():
    """Test message storage functionality"""
    print("\n🧪 Testing message storage...")
    
    try:
        from bot import HistoryBot
        
        # Create a test instance
        bot = HistoryBot()
        
        # Create test messages
        test_messages = [
            MockMessage("Hello, how are you?", "Alice"),
            MockMessage("I'm doing great! How about you?", "Bob"),
            MockMessage("Let's talk about the new project", "Charlie"),
            MockMessage("The weather is nice today", "David"),
            MockMessage("I love playing League of Legends", "Eve")
        ]
        
        # Store messages
        for msg in test_messages:
            bot.store_message(msg)
        
        print(f"✅ Stored {len(test_messages)} test messages")
        
        # Test search functionality
        search_query = "League of Legends"
        results = bot.search_messages(search_query)
        
        if results:
            print(f"✅ Search found {len(results)} relevant messages")
            print(f"   Top result: {results[0]['content'][:50]}...")
        else:
            print("❌ Search returned no results")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing message storage: {e}")
        return False

def test_json_persistence():
    """Test JSON file persistence"""
    print("\n🧪 Testing JSON persistence...")
    
    try:
        # Check if message_store.json exists
        if os.path.exists("message_store.json"):
            with open("message_store.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "messages" in data and len(data["messages"]) > 0:
                print(f"✅ JSON persistence works - {len(data['messages'])} messages stored")
                return True
            else:
                print("❌ JSON file exists but no messages found")
                return False
        else:
            print("❌ message_store.json file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing JSON persistence: {e}")
        return False

def cleanup_test_data():
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        if os.path.exists("message_store.json"):
            os.remove("message_store.json")
            print("✅ Cleaned up test data")
        else:
            print("ℹ️  No test data to clean up")
    except Exception as e:
        print(f"❌ Error cleaning up test data: {e}")

def main():
    """Run all tests"""
    print("🤖 HistoryBot Test Suite")
    print("=" * 40)
    
    # Check if API keys are available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("   Set it to run the full test suite")
        print("   export OPENAI_API_KEY=your_key_here")
        return
    
    tests = [
        test_embedding_functionality,
        test_message_storage,
        test_json_persistence
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HistoryBot is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Clean up test data
    cleanup_test_data()

if __name__ == "__main__":
    main() 