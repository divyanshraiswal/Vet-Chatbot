import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "Data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    try:
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents(texts, embeddings)

        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        db.save_local(DB_FAISS_PATH)
        print("Vector database created successfully!")
    except Exception as e:
        print(f"Error creating vector database: {e}")

if __name__ == "__main__":
    create_vector_db()
    