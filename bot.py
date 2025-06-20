import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

import discord
import numpy as np
import openai
from discord import app_commands
from discord.ext import commands

# Load API keys from environment variables
discord_token = os.getenv("DISCORD_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not discord_token:
    raise ValueError("DISCORD_TOKEN environment variable is required")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Set API key for the library
openai.api_key = openai_api_key

# Constants
DB_FILE = "data/message_store.db"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
RECALL_SIMILARITY_THRESHOLD = 0.75

class HistoryCog(commands.Cog):
    """Cog for storing and recalling message history using slash commands."""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        # Ensure the data directory exists before connecting to the DB
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        self.db = self.init_db()

    def init_db(self):
        """Initialize SQLite database and tables"""
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, content TEXT, author TEXT, author_id TEXT,
                channel TEXT, channel_id TEXT, guild TEXT, guild_id TEXT,
                timestamp TEXT, embedding TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS ask_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
                question TEXT, answer TEXT, timestamp TEXT
            )
        ''')
        conn.commit()
        return conn

    @commands.Cog.listener()
    async def on_ready(self):
        """Event fired when the bot is ready."""
        print(f"HistoryBot is ready as {self.bot.user}")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Event fired for every message. Stores non-command messages."""
        if message.author.bot:
            return
        await self.store_message(message)

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API (asynchronously)"""
        try:
            response = await self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2: return 0.0
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return dot / (norm1 * norm2)

    async def store_message(self, message: discord.Message):
        """Store message with embedding in SQLite"""
        try:
            if not message.content.strip(): return
            embedding = await self.get_embedding(message.content)
            if not embedding: return

            c = self.db.cursor()
            c.execute('''
                INSERT OR REPLACE INTO messages (
                    id, content, author, author_id, channel, channel_id, guild, guild_id, timestamp, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(message.id), message.content, message.author.display_name, str(message.author.id),
                message.channel.name, str(message.channel.id),
                message.guild.name if message.guild else "DM",
                str(message.guild.id) if message.guild else "DM",
                message.created_at.isoformat(), json.dumps(embedding)
            ))
            self.db.commit()
            print(f"Stored message from {message.author.display_name}: {message.content[:50]}...")
        except Exception as e:
            print(f"Error storing message: {e}")

    async def search_messages(self, query: str) -> List[Dict[str, Any]]:
        """Search messages using semantic similarity and a fixed threshold"""
        try:
            query_embedding = await self.get_embedding(query)
            if not query_embedding: return []
            
            # This part is synchronous, but fast enough not to need aiosqlite for now
            c = self.db.cursor()
            c.execute('SELECT * FROM messages')
            all_msgs = c.fetchall()
            
            similarities = [
                (self.cosine_similarity(query_embedding, json.loads(row[9])), row)
                for row in all_msgs if row[9]
            ]
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            filtered_rows = [row for sim, row in similarities if sim >= RECALL_SIMILARITY_THRESHOLD]
            
            return [{
                "id": r[0], "content": r[1], "author": r[2], "author_id": r[3],
                "channel": r[4], "channel_id": r[5], "guild": r[6], "guild_id": r[7],
                "timestamp": r[8], "embedding": json.loads(r[9])
            } for r in filtered_rows]
        except Exception as e:
            print(f"Error searching messages: {e}")
            return []

    def format_message_for_display(self, msg: Dict[str, Any]) -> str:
        """Format a stored message for display"""
        timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%m/%d/%Y [%I:%M %p]")
        author_display = f"**@{msg['author']}**"
        return f"{author_display}  _({timestamp})_:\n> {msg['content']}"

    def get_recent_channel_messages(self, channel_id: str, limit: int = 10) -> list:
        # ... (sync, fine for now)
        c = self.db.cursor()
        c.execute('SELECT author, content, timestamp FROM messages WHERE channel_id = ? ORDER BY timestamp DESC LIMIT ?', (channel_id, limit))
        return c.fetchall()[::-1]

    def get_user_ask_history(self, user_id: str, limit: int = 3) -> list:
        # ... (sync, fine for now)
        c = self.db.cursor()
        c.execute('SELECT question, answer, timestamp FROM ask_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
        return c.fetchall()[::-1]

    def store_ask_interaction(self, user_id: str, question: str, answer: str):
        # ... (sync, fine for now)
        c = self.db.cursor()
        c.execute('INSERT INTO ask_history (user_id, question, answer, timestamp) VALUES (?, ?, ?, ?)', (user_id, question, answer, datetime.utcnow().isoformat()))
        self.db.commit()

    @app_commands.command(name="ask", description="Ask me anything, with context from our chat.")
    @app_commands.describe(prompt="The question you want to ask.")
    async def ask_command(self, interaction: discord.Interaction, prompt: str):
        """Asks a question with chat history as context."""
        await interaction.response.defer()
        try:
            recent_msgs = self.get_recent_channel_messages(str(interaction.channel.id), limit=10)
            chat_history = "\n".join([f"{author}: {content}" for author, content, _ in recent_msgs])
            
            ask_history = self.get_user_ask_history(str(interaction.user.id), limit=3)
            ask_history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in ask_history])
            
            system_prompt = "You are a helpful assistant. Use the provided chat history and previous Q&A to answer the user's new question."
            full_prompt = (
                f"{system_prompt}\n\n--- RECENT CHAT HISTORY ---\n{chat_history}\n\n"
                f"--- PREVIOUS Q&A ---\n{ask_history_str}\n\n--- NEW QUESTION ---\n{prompt}"
            )
            
            response = await self.openai_client.chat.completions.create(
                model=CHAT_MODEL, messages=[{"role": "user", "content": full_prompt}]
            )
            answer = response.choices[0].message.content.strip()
            self.store_ask_interaction(str(interaction.user.id), prompt, answer)
            await interaction.followup.send(answer)
        except Exception as e:
            await interaction.followup.send(f"Error processing your request: {str(e)}")

    @app_commands.command(name="recall", description="Search through stored messages.")
    @app_commands.describe(query="What you want to search for.")
    async def recall_command(self, interaction: discord.Interaction, query: str):
        """Recalls relevant messages from history."""
        await interaction.response.defer()
        try:
            results = await self.search_messages(query)
            if not results:
                await interaction.followup.send(f"No relevant messages found (similarity threshold: {RECALL_SIMILARITY_THRESHOLD}).")
                return

            response_str = f"📝 Found {len(results)} relevant message(s):\n\n"
            for msg in results:
                response_str += f"{self.format_message_for_display(msg)}\n\n"

            if len(response_str) > 2000:
                # Simple chunking for long responses
                for i in range(0, len(response_str), 2000):
                    await interaction.followup.send(response_str[i:i+2000])
            else:
                await interaction.followup.send(response_str)
        except Exception as e:
            await interaction.followup.send(f"Error searching messages: {str(e)}")

    @app_commands.command(name="help", description="Shows the help message for HistoryBot.")
    async def help_command(self, interaction: discord.Interaction):
        """Displays the help message."""
        embed = discord.Embed(
            title="🤖 HistoryBot Help",
            description="I'm a bot that remembers your conversations and helps you recall information.",
            color=discord.Color.blue()
        )
        embed.add_field(
            name="`/ask <prompt>`",
            value="Ask me anything. I'll use recent chat history and your previous questions for context.",
            inline=False
        )
        embed.add_field(
            name="`/recall <query>`",
            value="Search through stored messages using natural language.",
            inline=False
        )
        embed.set_footer(text="I automatically store all non-command messages to build my memory.")
        await interaction.response.send_message(embed=embed)
        
    @commands.command()
    @commands.guild_only()
    @commands.is_owner()
    async def sync(self, ctx: commands.Context):
        """Syncs slash commands to the current guild."""
        synced = await ctx.bot.tree.sync(guild=ctx.guild)
        await ctx.send(f"Synced {len(synced)} commands to {ctx.guild.name}.")
        print(f"Synced {len(synced)} commands to {ctx.guild.name}.")


async def main():
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)
    async with bot:
        await bot.add_cog(HistoryCog(bot))
        await bot.start(discord_token)

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot is shutting down.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 
