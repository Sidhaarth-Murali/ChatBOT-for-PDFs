import os 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
import gradio as gr
import pandas as pd

user_query = ""
chat_history = [[]]
os.environ["HUGGINGFACE_API_TOKEN"] = "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"

with gr.Blocks() as app:
    chatbot = gr.Chatbot(label = "NLPchat")
    msg = gr.Textbox(placeholder="Ask me anything about nlp?")
    clear = gr.ClearButton([msg, chatbot])

    def chain():
        pdf_path = "/Users/sidhaarthmurali/Desktop/Exela-Internship/Task-1-ChatBOT-for-PDFs/FairnessInNLP.pdf"
        loader = PyPDFLoader(file_path=pdf_path)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 30)
        doc = text_splitter.split_documents(documents = document)

        embeddings =  HuggingFaceInstructEmbeddings(model_name="TheBloke/Llama-2-13B-chat-GGML")
        vectorstore = FAISS.from_documents(documents= doc, embedding= embeddings)
        vectorstore.save_local("BERT_learnings")

        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        new_vectorstore = FAISS.load_local("BERT_learnings", embeddings)
        qa = RetrievalQA.from_chain_type(llm = llm, chain_type = "refine", retriever = new_vectorstore.as_retriever())

        return qa
    
    link = chain()
        
    def respond(user_query, chat_history):
        bot_message = link.run(user_query)
        chat_history.append((user_query, bot_message))  
        return "", chat_history
    

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

app.launch()





