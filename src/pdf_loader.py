from langchain_community.document_loaders import PyPDFLoader
import os

def load_medical_pdfs(pdf_dir):
    documents = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            documents.extend(loader.load())
    return documents

if __name__ == "__main__":
    docs = load_medical_pdfs("data/medical_pdfs")
    print(f"Loaded {len(docs)} pages from medical PDFs")
