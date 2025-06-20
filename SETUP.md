# Setup Guide for HistoryBot

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Discord Bot Token** from Discord Developer Portal
3. **OpenAI API Key** from OpenAI Platform

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up API Keys

### Option A: Environment Variables (Recommended)

Create a `.env` file in the project directory:

```bash
# Discord Bot Token
DISCORD_TOKEN=your_discord_bot_token_here

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

Then install python-dotenv and modify the bot to load from .env:

```bash
pip install python-dotenv
```

Add this to the top of `bot_secure.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Option B: Direct Assignment (Less Secure)

Edit `bot.py` and replace the API keys directly:

```python
discord_token = "your_discord_bot_token_here"
openai.api_key = "your_openai_api_key_here"
```

## Step 3: Get Your API Keys

### Discord Bot Token

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Go to "Bot" section
4. Click "Add Bot"
5. Copy the token (click "Reset Token" if needed)
6. Enable "Message Content Intent" under "Privileged Gateway Intents"

### OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the API key

## Step 4: Invite Bot to Your Server

1. In Discord Developer Portal, go to "OAuth2" → "URL Generator"
2. Select scopes: `bot`, `applications.commands`
3. Select permissions: `Send Messages`, `Read Message History`, `Use Slash Commands`
4. Copy the generated URL and open it in your browser
5. Select your server and authorize the bot

## Step 5: Run the Bot

### Using the secure version (with environment variables):
```bash
python bot_secure.py
```

### Using the basic version (with hardcoded keys):
```bash
python bot.py
```

## Step 6: Test the Bot

In your Discord server, try these commands:

```
!help
!ask What's the weather like today?
!recall test message
```

## Troubleshooting

### "DISCORD_TOKEN environment variable is required"
- Make sure you've set the environment variable correctly
- Check that your `.env` file is in the same directory as the bot
- Verify the variable name matches exactly

### "OPENAI_API_KEY environment variable is required"
- Same as above, but for the OpenAI API key
- Make sure your OpenAI account has credits

### Bot doesn't respond to commands
- Check that the bot has the correct permissions in your server
- Verify that "Message Content Intent" is enabled in Discord Developer Portal
- Make sure the bot is online (check the console output)

### Embedding errors
- Check your OpenAI API key is valid
- Ensure you have sufficient credits in your OpenAI account
- The embedding model used is `text-embedding-ada-002`

## Security Notes

⚠️ **Never commit your API keys to version control!**

- Use environment variables or `.env` files
- Add `.env` to your `.gitignore` file
- Keep your API keys private and secure
- Rotate your keys regularly

## Next Steps

Once the bot is running, it will:
1. Automatically store all messages with embeddings
2. Allow you to search through conversation history
3. Provide AI-powered responses to your questions

The bot creates a `message_store.json` file that contains all stored messages and their embeddings. 