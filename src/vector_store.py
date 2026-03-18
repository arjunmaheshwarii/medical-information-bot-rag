<<<<<<< HEAD
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore
=======
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def create_vector_store(documents):
    """
    Create FAISS vector store from documents.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def search_vector_store(vector_store, query, k=5):
    """
    Perform similarity search with configurable results.
    """
    results = vector_store.similarity_search(query, k=k)
    
    # debug: print number of results
    print(f"Retrieved {len(results)} results")

    return results
>>>>>>> arjun-dev
