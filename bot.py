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
RECALL_SIMILARITY_THRESHOLD = 0.75

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
        """Initialize SQLite database and tables"""
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
        c.execute('''
            CREATE TABLE IF NOT EXISTS ask_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question TEXT,
                answer TEXT,
                timestamp TEXT
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
    
    def search_messages(self, query: str) -> List[Dict[str, Any]]:
        """Search messages using semantic similarity and a fixed threshold"""
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

            # Only return results with similarity above the threshold
            filtered = [msg for sim, msg in similarities if sim >= RECALL_SIMILARITY_THRESHOLD]
            return filtered

        except Exception as e:
            print(f"Error searching messages: {e}")
            return []
    
    def format_message_for_display(self, msg: Dict[str, Any]) -> str:
        """Format a stored message for display with clear MM/DD/YYYY date, 12-hour time with AM/PM, and @username."""
        timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%m/%d/%Y [%I:%M %p]")
        author_display = f"**@{msg['author']}**"
        return f"{author_display}  _({timestamp})_:\n> {msg['content']}"
    
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
            elif message.content == "!stats":
                await self.handle_stats_command(message)
            elif message.content == "!clear":
                await self.handle_clear_command(message)
            elif message.content == "!history":
                await self.handle_history_command(message)
    
    def get_recent_channel_messages(self, channel_id: str, limit: int = 10) -> list:
        """Fetch recent messages from a channel for context"""
        c = self.db.cursor()
        c.execute('''
            SELECT author, content, timestamp FROM messages
            WHERE channel_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (channel_id, limit))
        rows = c.fetchall()
        # Return in chronological order
        return rows[::-1]

    def get_user_ask_history(self, user_id: str, limit: int = 3) -> list:
        """Fetch recent ask Q&A pairs for a user"""
        c = self.db.cursor()
        c.execute('''
            SELECT question, answer, timestamp FROM ask_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        rows = c.fetchall()
        # Return in chronological order
        return rows[::-1]

    def store_ask_interaction(self, user_id: str, question: str, answer: str):
        c = self.db.cursor()
        c.execute('''
            INSERT INTO ask_history (user_id, question, answer, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, question, answer, datetime.utcnow().isoformat()))
        self.db.commit()

    async def handle_ask_command(self, message: discord.Message):
        """Handle !ask command with chat and ask history context"""
        try:
            prompt = message.content[len("!ask "):].strip()
            if not prompt:
                await message.channel.send("Please provide a prompt after !ask")
                return

            # Show typing indicator
            async with message.channel.typing():
                # Fetch recent chat history from the channel
                recent_msgs = self.get_recent_channel_messages(str(message.channel.id), limit=10)
                chat_history = "\n".join([
                    f"{author}: {content}" for author, content, _ in recent_msgs
                ])

                # Fetch previous ask Q&A for this user
                ask_history = self.get_user_ask_history(str(message.author.id), limit=3)
                ask_history_str = "\n".join([
                    f"Q: {q}\nA: {a}" for q, a, _ in ask_history
                ])

                # Build the system/context prompt
                system_prompt = "You are a helpful assistant. Here is the recent chat history and previous questions and answers. Use this context to answer the user's new question."
                full_prompt = (
                    f"{system_prompt}\n\n"
                    f"Recent chat history:\n{chat_history}\n\n"
                    f"Previous Q&A:\n{ask_history_str}\n\n"
                    f"New question: {prompt}"
                )

                # Call OpenAI Chat API
                response = openai.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": full_prompt}]
                )

                answer = response.choices[0].message.content.strip()

            await message.channel.send(answer)

            # Store the Q&A in ask_history
            self.store_ask_interaction(str(message.author.id), prompt, answer)

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

            # Show typing indicator while searching
            async with message.channel.typing():
                # Search for relevant messages
                results = self.search_messages(query)

            if not results:
                await message.channel.send(f"No relevant messages found (similarity threshold: {RECALL_SIMILARITY_THRESHOLD}).")
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
    
    async def handle_stats_command(self, message: discord.Message):
        """Handle !stats command"""
        try:
            stats = self.get_stats()
            
            stats_text = f"""
📊 **HistoryBot Statistics**

💬 **Messages Stored:** {stats['messages']:,}
❓ **Questions Asked:** {stats['questions']:,}
            """.strip()
            
            await message.channel.send(stats_text)
            
        except Exception as e:
            await message.channel.send(f"Error getting statistics: {str(e)}")

    async def handle_clear_command(self, message: discord.Message):
        """Handle !clear command - clear user's ask history and chat messages"""
        try:
            user_id = str(message.author.id)
            deleted_counts = self.clear_user_history(user_id)
            
            total_deleted = deleted_counts["ask_history"] + deleted_counts["messages"]
            
            if total_deleted > 0:
                response = f"🗑️ Cleared your history, {message.author.display_name}!\n"
                if deleted_counts["ask_history"] > 0:
                    response += f"• {deleted_counts['ask_history']} question(s) from ask history\n"
                if deleted_counts["messages"] > 0:
                    response += f"• {deleted_counts['messages']} message(s) from chat history"
                await message.channel.send(response)
            else:
                await message.channel.send(f"ℹ️ You don't have any history to clear, {message.author.display_name}.")
                
        except Exception as e:
            await message.channel.send(f"Error clearing your history: {str(e)}")

    async def handle_history_command(self, message: discord.Message):
        """Handle !history command - show user's message stats and recent messages"""
        try:
            user_id = str(message.author.id)
            stats = self.get_user_message_stats(user_id)
            
            # Build the response
            response = f"📝 **Message History for {message.author.display_name}**\n\n"
            response += f"💬 **Total Messages:** {stats['total_messages']:,}\n\n"
            
            if stats['recent_messages']:
                response += "**Recent Messages:**\n"
                for i, (content, timestamp) in enumerate(stats['recent_messages'], 1):
                    # Format timestamp
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime("%m/%d/%Y [%I:%M %p]")
                    
                    # Truncate long messages
                    display_content = content[:100] + "..." if len(content) > 100 else content
                    response += f"**{i}.** _{formatted_time}_\n> {display_content}\n\n"
            else:
                response += "No messages found."
            
            # Split if response is too long
            if len(response) > 2000:
                chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(response)
                
        except Exception as e:
            await message.channel.send(f"Error getting your message history: {str(e)}")

    async def handle_help_command(self, message: discord.Message):
        """Handle !help command"""
        help_text = """
🤖 **HistoryBot Commands:**

**!ask <prompt>** - Ask me anything using GPT-3.5-Turbo
Example: `!ask What's the weather like today?`

**!recall <query>** - Search through stored messages using natural language
Example: `!recall the time Kevin talked about Sion skin`

**!stats** - Show bot statistics (messages stored, questions asked)

**!history** - Show your message history (total count and recent messages)

**!clear** - Clear your own history (ask history and chat messages)

**!help** - Show this help message

💾 **Features:**
- Automatically stores all messages with embeddings
- Natural language search through conversation history
- Persistent memory across bot restarts (now using SQLite!)
- Context-aware AI responses with chat history
        """
        await message.channel.send(help_text)
    
    def get_stats(self) -> Dict[str, int]:
        """Get bot statistics from the database"""
        c = self.db.cursor()
        
        # Count total messages
        c.execute('SELECT COUNT(*) FROM messages')
        total_messages = c.fetchone()[0]
        
        # Count total questions asked
        c.execute('SELECT COUNT(*) FROM ask_history')
        total_questions = c.fetchone()[0]
        
        return {
            "messages": total_messages,
            "questions": total_questions
        }
    
    def clear_user_history(self, user_id: str) -> Dict[str, int]:
        """Clear both ask history and chat messages for a specific user"""
        c = self.db.cursor()
        
        # Clear ask history
        c.execute('DELETE FROM ask_history WHERE user_id = ?', (user_id,))
        ask_deleted = c.rowcount
        
        # Clear chat messages
        c.execute('DELETE FROM messages WHERE author_id = ?', (user_id,))
        messages_deleted = c.rowcount
        
        self.db.commit()
        
        return {
            "ask_history": ask_deleted,
            "messages": messages_deleted
        }
    
    def get_user_message_stats(self, user_id: str) -> Dict[str, Any]:
        """Get message statistics for a specific user"""
        c = self.db.cursor()
        
        # Count total messages
        c.execute('SELECT COUNT(*) FROM messages WHERE author_id = ?', (user_id,))
        total_messages = c.fetchone()[0]
        
        # Get recent messages
        c.execute('''
            SELECT content, timestamp FROM messages 
            WHERE author_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''', (user_id,))
        recent_messages = c.fetchall()
        
        return {
            "total_messages": total_messages,
            "recent_messages": recent_messages
        }
    
    def run(self):
        """Run the bot"""
        self.client.run(discord_token)

if __name__ == "__main__":
    bot = HistoryBot()
    bot.run() 
