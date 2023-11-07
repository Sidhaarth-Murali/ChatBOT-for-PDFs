import os 

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, VectorDBQA
from langchain import OpenAI

os.environ["OPENAI_API_KEY"] = "ccde586985ae480488b070af0321e017"

if __name__ == "__main__":
    pdf_path = "/Users/sidhaarthmurali/Desktop/Exela-Internship/Task-1-ChatBOT-for-PDFs/H2OGPT.pdf"
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

    user_query = ""

    while user_query.lower() != "thank you":
        user_query = input()
        result = qa.run(user_query)
        print(result)