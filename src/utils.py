import re
from langchain_community.document_loaders import WebBaseLoader

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters except common punctuation
    return text.strip()


def load_website_data(urls):
    """Load web documents and clean their content."""
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        return docs
    except Exception as e:
        print(f"⚠️ Error loading websites: {e}")
        return []