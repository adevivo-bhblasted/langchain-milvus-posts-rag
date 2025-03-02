import re
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader


def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters except common punctuation
    return text.strip()
    

# ---- Load and Process Documents ---- #
def load_and_process_documents(file_metadata: dict):
    """Loads and processes documents from different sources."""

    try:
        source_type = file_metadata.get("type")
        if source_type == "pdf":
            loader = PyPDFLoader(file_metadata.get("path"))
        elif source_type == "csv":
            loader = CSVLoader(file_metadata.get("path"))
        elif source_type == "website":
            loader = WebBaseLoader(file_metadata.get("urls"))
        else:
            print(f"⚠️ Unsupported document type: {source_type}")
            return []
    except Exception as e:
        print(f"⚠️ Error loading docs: {e}")
        return []

    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs
