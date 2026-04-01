from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=400, chunk_overlap=50):
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        documents: List of LangChain documents
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits raw text into chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():  # avoid empty chunks
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    print("Text chunking module ready.")