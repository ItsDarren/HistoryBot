import os
import json
import discord
import openai
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sqlite3
from collections import defaultdict

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
CHAT_MODEL = "gpt-3.5-turbo"
RATE_LIMIT = 3  # commands per minute
VERSION = "2.0.0"  # Bot version for tracking updates

class HistoryBot:
    def __init__(self):
        # Ensure the data directory exists
        os.makedirs("data", exist_ok=True)
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        self.db = self.init_db()
        self.command_cooldowns = defaultdict(list)  # Track command timestamps per user
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
                timestamp TEXT
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
    
    def store_message(self, message: discord.Message):
        """Store message in SQLite (no embeddings needed)"""
        try:
            if message.author.bot or not message.content.strip():
                return

            c = self.db.cursor()
            c.execute('''
                INSERT OR REPLACE INTO messages (
                    id, content, author, author_id, channel, channel_id, guild, guild_id, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(message.id),
                message.content,
                message.author.display_name,
                str(message.author.id),
                message.channel.name,
                str(message.channel.id),
                message.guild.name if message.guild else "DM",
                str(message.guild.id) if message.guild else "DM",
                message.created_at.isoformat()
            ))
            self.db.commit()
            print(f"Stored message from {message.author.display_name}: {message.content[:50]}...")

        except Exception as e:
            print(f"Error storing message: {e}")
    
    def ai_search_messages(self, query: str, messages: List[Dict[str, Any]]) -> List[tuple]:
        """Batch up to 200 messages and use OpenAI to determine relevance."""
        try:
            # Use the last 200 messages for context (increased from 100)
            recent_messages = messages[-200:]

            # Build a numbered list of messages for the prompt
            context_parts = []
            for idx, msg in enumerate(recent_messages, 1):
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%m/%d/%Y %I:%M %p")
                context_parts.append(f"{idx}. [{timestamp}] @{msg['author']}: {msg['content']}")
            context = "\n".join(context_parts)

            ai_prompt = f'''
You are a helpful assistant that understands natural language queries about chat history.

Given the user query: "{query}"

Here are some messages from a Discord chat:
{context}

For each message, determine if it's relevant to the user's query. Respond with a JSON array like:
[
  {{"index": 1, "relevant": true, "confidence": 0.95, "reason": "directly discusses the topic"}},
  {{"index": 2, "relevant": false, "confidence": 0.05, "reason": "unrelated conversation"}},
  ...
]

Guidelines:
- Only mark messages as relevant if they genuinely help answer the user's question
- Use confidence scores between 0 and 1 (higher = more relevant)
- Consider context, timing, and the actual content relevance
- Be selective - it's better to return fewer high-quality matches than many weak ones
'''

            response = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": ai_prompt}],
                temperature=0.1
            )

            ai_response = response.choices[0].message.content.strip()

            # Parse AI response
            try:
                if "```json" in ai_response:
                    json_start = ai_response.find("```json") + 7
                    json_end = ai_response.find("```", json_start)
                    json_str = ai_response[json_start:json_end].strip()
                elif "```" in ai_response:
                    json_start = ai_response.find("```") + 3
                    json_end = ai_response.find("```", json_start)
                    json_str = ai_response[json_start:json_end].strip()
                else:
                    json_str = ai_response
                ai_results = json.loads(json_str)

                results = []
                for ai_result in ai_results:
                    idx = ai_result.get("index")
                    relevant = ai_result.get("relevant", False)
                    confidence = ai_result.get("confidence", 0.5)
                    if relevant and 1 <= idx <= len(recent_messages):
                        msg = recent_messages[idx - 1]
                        results.append((confidence, msg))
                results.sort(key=lambda x: x[0], reverse=True)
                return results
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing AI response: {e}")
                return []
        except Exception as e:
            print(f"Error in AI search: {e}")
            return []
    
    def search_messages(self, query: str, date_filter: str = None) -> List[tuple]:
        """Search messages using only OpenAI-powered batching relevance."""
        try:
            c = self.db.cursor()
            sql_query = 'SELECT * FROM messages'
            params = []
            if date_filter:
                now = datetime.utcnow()
                if date_filter == "today":
                    start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                elif date_filter == "week":
                    start_date = now - timedelta(days=7)
                elif date_filter == "month":
                    start_date = now - timedelta(days=30)
                elif date_filter == "year":
                    start_date = now - timedelta(days=365)
                else:
                    date_filter = None
                if date_filter:
                    sql_query += ' WHERE timestamp >= ?'
                    params.append(start_date.isoformat())
            c.execute(sql_query, params)
            all_msgs = c.fetchall()
            if not all_msgs:
                return []
            messages = []
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
                    "timestamp": row[8]
                }
                messages.append(msg)
            relevant_messages = self.ai_search_messages(query, messages)
            return relevant_messages[:10]
        except Exception as e:
            print(f"Error searching messages: {e}")
            return []
    
    def format_message_for_display(self, msg: Dict[str, Any], confidence: int = None) -> str:
        """Format a stored message for display with clear MM/DD/YYYY date, 12-hour time with AM/PM, and @username."""
        timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%m/%d/%Y [%I:%M %p]")
        author_display = f"**@{msg['author']}**"
        
        if confidence is not None:
            return f"{author_display} **({confidence}% match)** _(on {timestamp})_:\n> {msg['content']}"
        else:
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
            if message.content.startswith("!ask ") or message.content.startswith("!a "):
                await self.handle_ask_command(message)
            elif message.content.startswith("!recall ") or message.content.startswith("!r "):
                await self.handle_recall_command(message)
            elif message.content == "!help" or message.content == "!h":
                await self.handle_help_command(message)
            elif message.content == "!version" or message.content == "!v":
                await self.handle_version_command(message)
            elif message.content == "!stats" or message.content == "!s":
                await self.handle_stats_command(message)
            elif message.content == "!history" or message.content == "!hist":
                await self.handle_history_command(message)
            elif message.content == "!clear" or message.content == "!c":
                await self.handle_clear_command(message)
    
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

    def is_user_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited (more than 3 commands per minute)"""
        now = datetime.utcnow()
        user_commands = self.command_cooldowns[user_id]
        
        # Remove commands older than 1 minute
        user_commands = [cmd_time for cmd_time in user_commands if now - cmd_time < timedelta(minutes=1)]
        self.command_cooldowns[user_id] = user_commands
        
        # Check if user has used 3 or more commands in the last minute
        return len(user_commands) >= RATE_LIMIT

    def add_user_command(self, user_id: str):
        """Add a new command timestamp for rate limiting"""
        self.command_cooldowns[user_id].append(datetime.utcnow())

    async def handle_ask_command(self, message: discord.Message):
        """Handle !ask command with chat and ask history context"""
        try:
            user_id = str(message.author.id)
            
            # Check rate limiting
            if self.is_user_rate_limited(user_id):
                await message.channel.send(f"⏰ Rate limit exceeded! You can only use {RATE_LIMIT} commands per minute. Please wait a moment before trying again.")
                return
            
            # Handle both !ask and !a prefixes
            if message.content.startswith("!ask "):
                prompt = message.content[len("!ask "):].strip()
            else:  # !a
                prompt = message.content[len("!a "):].strip()
                
            if not prompt:
                await message.channel.send("Please provide a prompt after !ask or !a")
                return

            # Add to rate limiting tracker
            self.add_user_command(user_id)

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
            user_id = str(message.author.id)
            
            # Check rate limiting
            if self.is_user_rate_limited(user_id):
                await message.channel.send(f"⏰ Rate limit exceeded! You can only use {RATE_LIMIT} commands per minute. Please wait a moment before trying again.")
                return
            
            # Handle both !recall and !r prefixes
            if message.content.startswith("!recall "):
                full_query = message.content[len("!recall "):].strip()
            else:  # !r
                full_query = message.content[len("!r "):].strip()
                
            if not full_query:
                await message.channel.send("Please provide a search query after !recall or !r")
                return

            # Parse date filter if present
            date_filter = None
            query = full_query
            
            if "date:" in full_query:
                parts = full_query.split(" ", 1)
                if len(parts) >= 2 and parts[0].startswith("date:"):
                    date_filter = parts[0][5:]  # Remove "date:" prefix
                    query = parts[1].strip()
                    
                    # Validate date filter
                    valid_filters = ["today", "week", "month", "year"]
                    if date_filter not in valid_filters:
                        await message.channel.send(f"Invalid date filter. Use: today, week, month, or year")
                        return

            # Add to rate limiting tracker
            self.add_user_command(user_id)

            search_msg = f"🔍 Searching for messages related to: *{query}*"
            if date_filter:
                search_msg += f" (from {date_filter})"
            await message.channel.send(search_msg)

            # Show typing indicator while searching
            async with message.channel.typing():
                # Search for relevant messages
                results = self.search_messages(query, date_filter)

            if not results:
                date_info = f" from {date_filter}" if date_filter else ""
                await message.channel.send(f"No relevant messages found{date_info}.")
                return

            # Format and send results with similarity scores
            date_info = f" (from {date_filter})" if date_filter else ""
            response = f"📝 Found {len(results)} relevant message(s){date_info}:\n\n"
            for i, (similarity, msg) in enumerate(results, 1):
                # Convert similarity to percentage
                confidence = int(similarity * 100)
                response += f"**{i}.** {self.format_message_for_display(msg, confidence)}\n\n"

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
            
            # Parse the full command to check for date filter
            full_command = message.content.strip()
            
            # Handle both !clear and !c prefixes
            if full_command.startswith("!clear "):
                args = full_command[len("!clear "):].strip()
            elif full_command.startswith("!c "):
                args = full_command[len("!c "):].strip()
            else:
                # No arguments, clear all history
                args = ""
            
            date_filter = None
            
            # Check if date filter is specified
            if args.startswith("date:"):
                parts = args.split(" ", 1)
                if len(parts) >= 1:
                    date_filter = parts[0][5:]  # Remove "date:" prefix
                    
                    # Validate date filter
                    valid_filters = ["today", "week", "month", "year"]
                    if date_filter not in valid_filters:
                        await message.channel.send(f"Invalid date filter. Use: today, week, month, or year")
                        return
            
            deleted_counts = self.clear_user_history(user_id, date_filter)
            
            total_deleted = deleted_counts["ask_history"] + deleted_counts["messages"]
            date_info = f" from {date_filter}" if date_filter else ""
            
            if total_deleted > 0:
                response = f"🗑️ Cleared your history{date_info}, {message.author.display_name}!\n"
                if deleted_counts["ask_history"] > 0:
                    response += f"• {deleted_counts['ask_history']} question(s) from ask history\n"
                if deleted_counts["messages"] > 0:
                    response += f"• {deleted_counts['messages']} message(s) from chat history"
                await message.channel.send(response)
            else:
                await message.channel.send(f"ℹ️ You don't have any history{date_info} to clear, {message.author.display_name}.")
                
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
**HistoryBot Commands:**

**!ask / !a <prompt>** - Ask me anything using GPT-3.5-Turbo
Example: `!ask What is 9+1?` or `!a What is 9+1?`

**!recall / !r <query>** - Search through stored messages using natural language
Example: `!recall when we talked about genshin impact` or `!r when we talked about genshin impact`
    **Date Filtering:** Add `date:today` (or week/month/year) to search specific time periods
    Example: `!recall date:today jordan` or `!r date:week project discussion`

**!stats / !s** - Show bot statistics (messages stored, questions asked)

**!history / !hist** - Show your message history (total count and recent messages)

**!clear / !c** - Clear your own history (ask history and chat messages)
    **Date Filtering:** Add `date:today` (or week/month/year) to clear specific time periods
    Example: `!clear date:today` or `!c date:week`

**!version / !v** - Show bot version and update information

**!help / !h** - Show this help message

💾 **Features:**
- Automatically stores all messages
- AI-powered natural language search through conversation history
- Persistent memory across bot restarts (using SQLite!)
- Context-aware AI responses with chat history
- Command aliases for faster typing
- Date filtering for targeted searches and history clearing
        """
        await message.channel.send(help_text)

    async def handle_version_command(self, message: discord.Message):
        """Handle !version command"""
        version_text = f"""
🤖 **HistoryBot v{VERSION}**

**Latest Updates:**
• Switched to 100% OpenAI-powered search (no more embeddings!)
• Increased search batch size to 200 messages per API call
• Improved AI relevance detection
• Removed numpy dependency for cleaner setup

**Technical Details:**
• Model: {CHAT_MODEL}
• Database: SQLite
• Search Method: OpenAI GPT-3.5-Turbo
• Rate Limit: {RATE_LIMIT} commands/minute

*Use `!help` for command list*
        """.strip()
        await message.channel.send(version_text)
    
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
    
    def clear_user_history(self, user_id: str, date_filter: str = None) -> Dict[str, int]:
        """Clear both ask history and chat messages for a specific user, optionally filtered by date"""
        c = self.db.cursor()
        
        # Build date filter if specified
        date_condition = ""
        params = [user_id]
        
        if date_filter:
            now = datetime.utcnow()
            if date_filter == "today":
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif date_filter == "week":
                start_date = now - timedelta(days=7)
            elif date_filter == "month":
                start_date = now - timedelta(days=30)
            elif date_filter == "year":
                start_date = now - timedelta(days=365)
            else:
                # Invalid date filter, ignore it
                date_filter = None
            
            if date_filter:
                date_condition = " AND timestamp >= ?"
                params.append(start_date.isoformat())
        
        # Clear ask history
        c.execute(f'DELETE FROM ask_history WHERE user_id = ?{date_condition}', params)
        ask_deleted = c.rowcount
        
        # Clear chat messages (reset params for messages table)
        params = [user_id]
        if date_filter:
            params.append(start_date.isoformat())
        c.execute(f'DELETE FROM messages WHERE author_id = ?{date_condition}', params)
        messages_deleted = c.rowcount
        
        self.db.commit()
        
        return {
            "ask_history": ask_deleted,
            "messages": messages_deleted,
            "date_filter": date_filter
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
