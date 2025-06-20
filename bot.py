import os
import json
import discord
import openai
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Load API keys from environment variables
discord_token = os.getenv("DISCORD_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not discord_token:
    raise ValueError("DISCORD_TOKEN environment variable is required")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Constants
MESSAGE_STORE_FILE = "message_store.json"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
MAX_RECALL_RESULTS = 5

class HistoryBot:
    def __init__(self):
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        self.message_store = self.load_message_store()
        self.setup_events()
    
    def load_message_store(self) -> Dict[str, Any]:
        """Load message store from JSON file"""
        try:
            if os.path.exists(MESSAGE_STORE_FILE):
                with open(MESSAGE_STORE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"messages": []}
        except Exception as e:
            print(f"Error loading message store: {e}")
            return {"messages": []}
    
    def save_message_store(self):
        """Save message store to JSON file"""
        try:
            with open(MESSAGE_STORE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.message_store, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving message store: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API"""
        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def store_message(self, message: discord.Message):
        """Store message with embedding"""
        try:
            # Skip bot messages and empty messages
            if message.author.bot or not message.content.strip():
                return
            
            # Get embedding for message content
            embedding = self.get_embedding(message.content)
            if not embedding:
                return
            
            # Create message record
            message_record = {
                "id": str(message.id),
                "content": message.content,
                "author": message.author.display_name,
                "author_id": str(message.author.id),
                "channel": message.channel.name,
                "channel_id": str(message.channel.id),
                "guild": message.guild.name if message.guild else "DM",
                "guild_id": str(message.guild.id) if message.guild else "DM",
                "timestamp": message.created_at.isoformat(),
                "embedding": embedding
            }
            
            # Add to store
            self.message_store["messages"].append(message_record)
            
            # Save to file
            self.save_message_store()
            
            print(f"Stored message from {message.author.display_name}: {message.content[:50]}...")
            
        except Exception as e:
            print(f"Error storing message: {e}")
    
    def search_messages(self, query: str, limit: int = MAX_RECALL_RESULTS) -> List[Dict[str, Any]]:
        """Search messages using semantic similarity"""
        try:
            # Get embedding for query
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Calculate similarities
            similarities = []
            for msg in self.message_store["messages"]:
                if "embedding" in msg:
                    similarity = self.cosine_similarity(query_embedding, msg["embedding"])
                    similarities.append((similarity, msg))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Return top results
            return [msg for _, msg in similarities[:limit]]
            
        except Exception as e:
            print(f"Error searching messages: {e}")
            return []
    
    def format_message_for_display(self, msg: Dict[str, Any]) -> str:
        """Format a stored message for display"""
        timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M")
        return f"**{msg['author']}** ({timestamp}): {msg['content']}"
    
    def setup_events(self):
        """Setup Discord event handlers"""
        
        @self.client.event
        async def on_ready():
            print(f"HistoryBot is ready as {self.client.user}")
            print(f"Loaded {len(self.message_store['messages'])} stored messages")
        
        @self.client.event
        async def on_message(message):
            # Ignore messages from ourselves
            if message.author == self.client.user:
                return
            
            # Store all messages (except commands)
            if not message.content.startswith("!"):
                self.store_message(message)
            
            # Handle commands
            if message.content.startswith("!ask "):
                await self.handle_ask_command(message)
            elif message.content.startswith("!recall "):
                await self.handle_recall_command(message)
            elif message.content == "!help":
                await self.handle_help_command(message)
    
    async def handle_ask_command(self, message: discord.Message):
        """Handle !ask command"""
        try:
            prompt = message.content[len("!ask "):].strip()
            if not prompt:
                await message.channel.send("Please provide a prompt after !ask")
                return
            
            # Call OpenAI Chat API
            response = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response.choices[0].message.content.strip()
            await message.channel.send(answer)
            
        except Exception as e:
            await message.channel.send(f"Error processing your request: {str(e)}")
    
    async def handle_recall_command(self, message: discord.Message):
        """Handle !recall command"""
        try:
            query = message.content[len("!recall "):].strip()
            if not query:
                await message.channel.send("Please provide a search query after !recall")
                return
            
            await message.channel.send(f"🔍 Searching for messages related to: *{query}*")
            
            # Search for relevant messages
            results = self.search_messages(query)
            
            if not results:
                await message.channel.send("No relevant messages found.")
                return
            
            # Format and send results
            response = f"📝 Found {len(results)} relevant message(s):\n\n"
            for i, msg in enumerate(results, 1):
                response += f"**{i}.** {self.format_message_for_display(msg)}\n\n"
            
            # Split if response is too long
            if len(response) > 2000:
                chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(response)
                
        except Exception as e:
            await message.channel.send(f"Error searching messages: {str(e)}")
    
    async def handle_help_command(self, message: discord.Message):
        """Handle !help command"""
        help_text = """
🤖 **HistoryBot Commands:**

**!ask <prompt>** - Ask me anything using GPT-3.5-Turbo
Example: `!ask What's the weather like today?`

**!recall <query>** - Search through stored messages using natural language
Example: `!recall the time Kevin talked about Sion skin`

**!help** - Show this help message

💾 **Features:**
- Automatically stores all messages with embeddings
- Natural language search through conversation history
- Persistent memory across bot restarts
        """
        await message.channel.send(help_text)
    
    def run(self):
        """Run the bot"""
        self.client.run(discord_token)

if __name__ == "__main__":
    bot = HistoryBot()
    bot.run() 