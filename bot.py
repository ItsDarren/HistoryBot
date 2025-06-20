#testing memory

import os
import json
import discord
import openai
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import sqlite3

# Load API keys from environment variables
discord_token = os.getenv("DISCORD_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not discord_token:
    raise ValueError("DISCORD_TOKEN environment variable is required")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Constants
DB_FILE = "data/message_store.db"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
MAX_RECALL_RESULTS = 5

class HistoryBot:
    def __init__(self):
        # Ensure the data directory exists
        os.makedirs("data", exist_ok=True)
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        self.db = self.init_db()
        self.setup_events()
    
    def init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                content TEXT,
                author TEXT,
                author_id TEXT,
                channel TEXT,
                channel_id TEXT,
                guild TEXT,
                guild_id TEXT,
                timestamp TEXT,
                embedding TEXT
            )
        ''')
        conn.commit()
        return conn
    
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
        """Store message with embedding in SQLite"""
        try:
            if message.author.bot or not message.content.strip():
                return

            embedding = self.get_embedding(message.content)
            if not embedding:
                return

            c = self.db.cursor()
            c.execute('''
                INSERT OR REPLACE INTO messages (
                    id, content, author, author_id, channel, channel_id, guild, guild_id, timestamp, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(message.id),
                message.content,
                message.author.display_name,
                str(message.author.id),
                message.channel.name,
                str(message.channel.id),
                message.guild.name if message.guild else "DM",
                str(message.guild.id) if message.guild else "DM",
                message.created_at.isoformat(),
                json.dumps(embedding)
            ))
            self.db.commit()
            print(f"Stored message from {message.author.display_name}: {message.content[:50]}...")

        except Exception as e:
            print(f"Error storing message: {e}")
    
    def search_messages(self, query: str, limit: int = MAX_RECALL_RESULTS) -> List[Dict[str, Any]]:
        """Search messages using semantic similarity"""
        try:
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []

            c = self.db.cursor()
            c.execute('SELECT * FROM messages')
            all_msgs = c.fetchall()

            similarities = []
            for row in all_msgs:
                msg = {
                    "id": row[0],
                    "content": row[1],
                    "author": row[2],
                    "author_id": row[3],
                    "channel": row[4],
                    "channel_id": row[5],
                    "guild": row[6],
                    "guild_id": row[7],
                    "timestamp": row[8],
                    "embedding": json.loads(row[9])
                }
                similarity = self.cosine_similarity(query_embedding, msg["embedding"])
                similarities.append((similarity, msg))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Only return results with similarity above a threshold (e.g., 0.7)
            threshold = 0.7
            filtered = [msg for sim, msg in similarities if sim >= threshold]

            # Return up to 'limit' results
            return filtered[:limit]

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
