from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    """
    Create FAISS vector store from document chunks.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


def search_vector_store(vector_store, query, k=5):
    """
    Perform similarity search with configurable results.
    
    Args:
        vector_store: FAISS vector store
        query (str): search query
        k (int): number of results
        
    Returns:
        List of relevant documents
    """
    results = vector_store.similarity_search(query, k=k)

    print(f"Retrieved {len(results)} results")

    return results