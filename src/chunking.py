from langchain_text_splitters import RecursiveCharacterTextSplitter

<<<<<<< HEAD

def chunk_text(text, chunk_size=400, overlap=100):
    """
    Splits LangChain documents into smaller chunks.
    """
=======
def chunk_documents(documents, chunk_size=400, chunk_overlap=50):
>>>>>>> arjun-dev
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

<<<<<<< HEAD

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits raw text into chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # avoid empty chunks
        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample_text = "This is a sample medical text for testing chunking."
    chunks = chunk_text(sample_text)
    print(f"Generated {len(chunks)} chunks.")
=======
if __name__ == "__main__":
    print("Text chunking module ready.")
>>>>>>> arjun-dev
