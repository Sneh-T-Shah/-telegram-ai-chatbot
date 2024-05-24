# Telegram AI Chatbot

This project creates an AI chatbot for Telegram channels using the OpenAI API and the Retrieval-Augmented Generation (RAG) approach. The chatbot can retrieve relevant information from a set of documents or URLs to provide informed responses to user queries on a specific topic. It also includes a web scraper that can automatically scrape data from any website and generate text files for the RAG process.

## Prerequisites

- Python 3.x
- Gemmini api key
- Telegram Bot API token and username

## Installation

1. Clone the repository:

```
git clone https://github.com/your-repo/telegram-ai-chatbot.git
```

2. Install the required Python packages:

```
pip install -r requirements.txt
```

3. Set your Gemmini API key as an environment variable:

```
export os.environ['GEMINI_API_KEY'] = 'Your api key here'
```

4. Open the `main.py` file and replace the following placeholders with your actual values:

```python
TOKEN: Final = "your_telegram_bot_token"
BOT_USERNAME: Final = "your_bot_username"
```

5. In the `start_command` function, update the message to provide information about the topic your chatbot will cover:

```python
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me. You can chat with me on the [topic] topic!')
```

6. In the `llms.py` file, update the `urls` list with the URLs you want to scrape and use as the knowledge base for the chatbot.

7. You can get the Token and bot username from the botfather bot of telegram you can find it's guidelines from online sources.
## Usage

1. Run the `main.py` script:

```
python main.py
```

2. The bot will start polling for updates from Telegram.

3. You can interact with the bot by sending messages to it in Telegram channels or private chats. The bot will respond with relevant information based on the knowledge base created from the specified URLs and the topic you defined.

## Code Explanation

- The `get_url_text_and_make_pdf` function in `llms.py` scrapes the text content from a given URL using the `goose3` library and saves it as a text file.
- The `make_embeddings` function in `llms.py` loads the text files, splits them into smaller chunks, creates embeddings using the OpenAI Embeddings model, and stores the embeddings in a FAISS vector store.
- The `handle_response` function in `main.py` processes the user's message, retrieves relevant information from the knowledge base using the `get_response_from_query` function from the `llms.py` module, and returns the generated response.
- The `handle_message` function in `main.py` is a handler for text messages received by the bot. It checks if the message was sent in a group or private chat, and if the bot's username is mentioned in the group message. It then passes the user's message to the `handle_response` function and sends the generated response back to the user.
- The `error` function in `main.py` is an error handler that logs any errors that occur during the bot's operation.
- The `Application` object from the `python-telegram-bot` library is used to create the Telegram bot, add handlers for commands and messages, and start polling for updates.

## Production Readiness

This codebase is designed to be production-ready, meaning you can clone the repository, install the required dependencies, and deploy the chatbot with minimal modifications. However, it's essential to follow best practices for deploying and securing your application in a production environment, such as setting up appropriate logging, monitoring, and scaling mechanisms.

## Requirements

The project's dependencies are listed in the `requirements.txt` file. You can install them using the following command:

```
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
