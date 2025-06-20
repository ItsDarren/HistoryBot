# HistoryBot 🤖

A personal Discord bot designed to act like a memory-keeping assistant. It uses OpenAI GPT API and embeddings to store, understand, and recall user messages from Discord servers.

## Features

### Core Commands

- **`!ask <prompt>`** - Directly sends a prompt to GPT-3.5-Turbo and returns the response
- **`!recall <query>`** - Search through stored messages using natural language
- **`!help`** - Show available commands and features

### Memory Features

- **Automatic Message Storage**: All messages (except commands) are automatically stored with vector embeddings
- **Natural Language Search**: Find relevant messages by describing them in natural language
- **Persistent Memory**: Message data is saved to `message_store.json` and persists across bot restarts
- **Semantic Search**: Uses OpenAI embeddings and cosine similarity for intelligent message retrieval

## Example Usage

```
!ask What's the weather like today?
!recall the time Kevin talked about Sion skin
!recall when we discussed the new project
!help
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Keys**:
   - Get your Discord bot token from the [Discord Developer Portal](https://discord.com/developers/applications)
   - Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Update the API keys in `bot.py`

3. **Run the Bot**:
   ```bash
   python bot.py
   ```

## Technical Details

### Dependencies
- `discord.py` - Discord bot framework
- `openai` - OpenAI API client
- `numpy` - Numerical computations for similarity calculations

### Storage
- Messages are stored in `message_store.json` with the following structure:
  ```json
  {
    "messages": [
      {
        "id": "message_id",
        "content": "message content",
        "author": "author_name",
        "author_id": "author_id",
        "channel": "channel_name",
        "channel_id": "channel_id",
        "guild": "server_name",
        "guild_id": "server_id",
        "timestamp": "2024-01-01T12:00:00",
        "embedding": [0.1, 0.2, ...]
      }
    ]
  }
  ```

### Search Algorithm
- Uses OpenAI's `text-embedding-ada-002` model for generating embeddings
- Implements cosine similarity for semantic search
- Returns top 5 most relevant messages by default

## Future Enhancements

- [ ] Move to FAISS or vector database for better performance
- [ ] Add message filtering by date, author, or channel
- [ ] Implement conversation threading
- [ ] Add message summarization features
- [ ] Support for file attachments and media

## Security Notes

⚠️ **Important**: The current implementation has API keys hardcoded in the source code. For production use:

1. Use environment variables for API keys
2. Add proper error handling and rate limiting
3. Implement user authentication and permissions
4. Consider data privacy and GDPR compliance

## License

This project is for personal use. Please respect Discord's Terms of Service and OpenAI's usage policies. 