from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

<<<<<<< HEAD
from pdf_loader import load_medical_pdfs
from chunking import chunk_documents
from vector_store import create_vector_store


def run_rag_query(query: str):
=======
from src.pdf_loader import load_medical_pdfs
from src.chunking import chunk_documents
from src.vector_store import create_vector_store


def run_rag_query(query: str):
    """
    Runs a full RAG pipeline:
    - Load PDFs
    - Chunk documents
    - Store embeddings
    - Retrieve relevant context
    - Generate answer
    """

>>>>>>> arjun-dev
    # Load PDFs
    docs = load_medical_pdfs("data/medical_pdfs")

    # Chunk documents
    chunks = chunk_documents(docs)

    # Vector store
    vectorstore = create_vector_store(chunks)
<<<<<<< HEAD
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
=======
    retriever = get_retriever(vectorstore)
>>>>>>> arjun-dev

    # Retrieve context
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

<<<<<<< HEAD
    # Prompt
=======
    # Prompt template
>>>>>>> arjun-dev
    prompt = PromptTemplate.from_template(
        """
You are a medical assistant.
Answer using ONLY the context below.

Context:
{context}

Question:
{question}
"""
    )

<<<<<<< HEAD
    # LLM (FREE)
    pipe = pipeline(
        "text-generation",
=======
    # LLM (FREE HuggingFace model)
    pipe = pipeline(
        "text2text-generation",  # better for flan-t5
>>>>>>> arjun-dev
        model="google/flan-t5-base",
        max_new_tokens=200
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Chain
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response


if __name__ == "__main__":
    print("RAG query pipeline ready.")
<<<<<<< HEAD
    answer = run_rag_query("What is common infection?")
=======

    query = "What is a common infection?"
    answer = run_rag_query(query)

    print("\nQuestion:", query)
>>>>>>> arjun-dev
    print("\nAnswer:\n", answer)