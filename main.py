from typing import Final
from llms import make_embeddings,get_response_from_query
TOKEN: Final = "your token here"
BOT_USERNAME: Final = "bot username here"

import nest_asyncio
nest_asyncio.apply()


from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters,ContextTypes

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me.You can chat with me on the ..... topic!')

def handle_response(text: str) -> str:
    text = text.lower()
    response, docs = get_response_from_query(db, text, chain, k=20)
    return response
        
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text = str = update.message.text

    print(f'User ({update.message.chat.id})in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:   
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)
    
    print('Bot:', response)
    await update.message.reply_text(response)
    
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
        print(f'Update {update} caused error {context.error}')
        

chain,db = make_embeddings()
    
if __name__ == '__main__':
    print('Starting bot...')
    app =  Application.builder().token(TOKEN).http_version("1.1").build()

# Add command handlers
    app.add_handler(CommandHandler("start", start_command))
    
    # Message handler for all text messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
  
    # Error handler
    app.add_error_handler(error)
    print('Polling...')
    app.run_polling()