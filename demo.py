import os 
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, VectorDBQA
from langchain import OpenAI
import gradio as gr

os.environ["OPENAI_API_KEY"] = "ssk-ifp0KVix9I63ZiGHjvajT3BlbkFJgutpZyqHp81CXWl8wc8U"
user_query = ""
chat_history = [[]]

with gr.Blocks() as app:
    chatbot = gr.Chatbot(label = "QuantumML chat")
    msg = gr.Textbox(placeholder="Ask me anything about Quantum Machine Learning?")
    clear = gr.ClearButton([msg, chatbot])

    def chain():
        pdf_path = "/Users/sidhaarthmurali/Desktop/Exela-Internship/Task-1-ChatBOT-for-PDFs/FairnessInNLP.pdf"
        loader = PyPDFLoader(file_path=pdf_path)
        document = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 30)
        doc = text_splitter.split_documents(documents = document)

        embeddings =  OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents= doc, embedding= embeddings)
        vectorstore.save_local("QML-Learnings")

        llm = OpenAI()
        new_vectorstore = FAISS.load_local("QML-Learnings", embeddings)
        qa = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = new_vectorstore.as_retriever())

        return qa
    
    link = chain()
        
    def respond(user_query, chat_history):
        bot_message = link.run(user_query)
        chat_history.append((user_query, bot_message))
        return "", chat_history
    

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

app.launch()





