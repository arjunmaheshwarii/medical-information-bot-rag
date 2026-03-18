# Added basic PDF loader structure
from langchain_community.document_loaders import PyPDFLoader
import os


def load_medical_pdfs(pdf_dir):
    documents = []

    # check if directory exists
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, file)

            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return documents


if __name__ == "__main__":
    pdf_directory = "data/medical_pdfs"
    docs = load_medical_pdfs(pdf_directory)
    print(f"Loaded {len(docs)} pages from medical PDFs")