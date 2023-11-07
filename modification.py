import os 
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
import gradio as gr


OPENAI_API_TYPE = "Azure"
OPENAI_API_VERSION = "2023-05-15"
OPENAI_API_BASE = "https://pdf-chatbot.openai.azure.com/"
OPENAI_API_KEY = "ccde586985ae480488b070af0321e017"
DEPLOYMENT_NAME = "gpt-35-turbo"
DEPLOYMENT_ID = "text-embedding-ada-002"

os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
load_dotenv()



user_query = ""
chat_history = [[]]


with gr.Blocks() as app:
    chatbot = gr.Chatbot(label = "NLP chat")
    msg = gr.Textbox(placeholder="Ask me something?")
    clear = gr.ClearButton([msg, chatbot])

    def chain():
        pdf_path = "/Users/sidhaarthmurali/Desktop/Exela-Internship/Task-1-ChatBOT-for-PDFs/FairnessInNLP.pdf"
        loader = PyPDFLoader(file_path=pdf_path)
        document = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 30)
        doc = text_splitter.split_documents(documents = document)

        embeddings = OpenAIEmbeddings(engine="text-embedding-ada-002")
        vectorstore = Chroma.from_documents(documents= doc, embedding= embeddings)

        llm = AzureOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", temperature = 0.8)
        qa = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = vectorstore.as_retriever())

        return qa
    
    link = chain()
        
    def respond(user_query, chat_history):
        bot_message = link.run(user_query)
        chat_history.append((user_query, bot_message))
        return "", chat_history
    

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

app.launch()





