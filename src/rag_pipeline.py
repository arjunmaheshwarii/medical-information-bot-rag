from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

    # Load PDFs
    docs = load_medical_pdfs("data/medical_pdfs")

    # Chunk documents
    chunks = chunk_documents(docs)

    # Vector store
    vectorstore = create_vector_store(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Retrieve context
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Prompt template
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

    # LLM (FREE HuggingFace model)
    pipe = pipeline(
        "text2text-generation",  # better for flan-t5
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

    query = "What is a common infection?"
    answer = run_rag_query(query)

    print("\nQuestion:", query)
    print("\nAnswer:\n", answer)