from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def create_vector_store(documents):
    """
    Create FAISS vector store from documents.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def search_vector_store(vector_store, query):
    """
    Perform similarity search.
    """
    return vector_store.similarity_search(query, k=3)