import os 
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

if __name__ == "__main__":
    print("Hello pyPDF!")

    pdf_path = "/Users/sidhaarthmurali/Desktop/Exela-Internship/Task-1-ChatBOT-for-PDFs/H2OGPT.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc = text_splitter.split_documents(documents=document)

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_documents(documents=doc, embedding=embeddings)

    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":1024})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    user_query = ""

    while True:
        user_query = input("Enter your query (type 'thank you' to exit): ")
        if user_query.lower() == "thank you":
            break
        result = qa.run(user_query)
        print(result)
