from typing import List, Optional
from langchain_milvus import Milvus
import numpy as np
from ..commons.constants.milvus import Constants as MilvusConstants
from langchain_core.documents import Document

from ..utils import load_and_process_documents
from ..commons.ollama.utils import ollama_embeddings

def get_vectorstore():
    return Milvus(
        embedding_function=ollama_embeddings,
        connection_args={"host": MilvusConstants.HOST_NAME, "port": MilvusConstants.PORT},
        collection_name=MilvusConstants.COLLECTION_NAME,
        auto_id=True
    )


def create_document(text: str) -> Document:
    return Document(page_content=text, metadata={"id": text[:36]})


def add_texts_to_vectorstore(texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
    """
    Aggiunge testi al vector store

    Args:
        texts (List[str]): Lista di testi da aggiungere
        metadatas (Optional[List[dict]]): Metadati opzionali per ogni testo

    Returns:
        List[str]: Lista degli ID dei documenti inseriti
    """
    try:
        print(f"Aggiunta di {len(texts)} documenti...")
        vectorstore = get_vectorstore()
        ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print("✓ Documenti aggiunti con successo")
        return ids
    except Exception as e:
        print(f"✗ Errore nell'aggiunta dei documenti: {e}")
        raise e


def add_document_from_text(text: str):
    vectorstore = get_vectorstore()
    document = create_document(text)
    vectorstore.add_documents([document])


def add_document_from_import(file_metadata: dict):
    vectorstore = get_vectorstore()
    documents = load_and_process_documents(file_metadata)
    vectorstore.add_documents(documents)

