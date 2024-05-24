import os
from dotenv import load_dotenv
import openai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
from langchain_community.document_loaders import TextLoader
import chainlit as cl
import goose3
from langchain.memory import ConversationBufferMemory



goose = goose3.Goose()

def get_url_text_and_make_pdf(url: str,i:int):
    # Get the text from the URL
    article = goose.extract(url=url)
    text = article.cleaned_text
    with open(f"pdfs/{i}.txt", "w",encoding="utf-8") as f:
        f.write(text)



load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create OpenAI Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")


def make_embeddings():
    urls = ["https://screenrant.com/pokemon-facts-trivia-things-you-never-knew-brock/","https://pokemon.fandom.com/wiki/Brock_(anime)","https://bulbapedia.bulbagarden.net/wiki/Brock_(anime)"]
    i = 0
    for url in urls:
        get_url_text_and_make_pdf(url,i)
        i+=1
    pdf_folder = "pdfs/"
    documents = []
    for pdf_file in Path(pdf_folder).glob("*.txt"):
        loader = TextLoader(pdf_folder + str(pdf_file.name),encoding = 'UTF-8')
        text = loader.load()
        documents.extend(text)

    
    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    # Create FAISS vector store
    db = FAISS.from_documents(texts, embeddings)

    # Create the chat model adujusting the temperature to 0.3 for more accurate responses can reduce the temperature for less creative response
    os.environ['GEMINI_API_KEY'] = '...........'
    api_key = os.getenv('GEMINI_API_KEY')

    
    chat = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)


    
    # Create the LLM chain with a custom prompt
    prompt = PromptTemplate(
        input_variables=["docs" ,"chat_history","human_input"],
        template=("""
        You are a interactive chatbot that can answer questions about Brok a character from pokemon series.
        here is the chat history so far:
        {chat_history}
        Answer the following question based on the context of chat history if it is related: {human_input}
        By searching the following text: {docs}
        Only use the factual information from the text to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        """)
    )
    memory=ConversationBufferMemory(
    memory_key="chat_history",
    input_key="human_input"
    )
    chain = LLMChain(llm=chat, prompt=prompt,memory=memory,verbose=False)
    return chain,db


def get_response_from_query(db, query, chain, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    print(docs_page_content)
    response = chain.predict(docs=docs_page_content,human_input=query)
    return response, docs
