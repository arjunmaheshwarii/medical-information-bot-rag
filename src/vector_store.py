from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def create_vector_store(documents):
    """
    Create FAISS vector store from documents.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store