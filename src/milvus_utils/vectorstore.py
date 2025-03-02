from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
from commons.constants.milvus import Constants as MilvusConstants
from commons.constants.ollama import Constants as OllamaConstants
from langchain_core.documents import Document

from ..utils import load_and_process_documents

def get_vectorstore():
    embeddings = OllamaEmbeddings(
        model=OllamaConstants.OLLAMA_EMBEDDING_MODEL,
        chunk_size=300
    )
    return Milvus(
        embedding_function=embeddings,
        connection_args={"host": MilvusConstants.HOST_NAME, "port": MilvusConstants.PORT},
        collection_name=MilvusConstants.COLLECTION_NAME,
    )


def create_document(text: str) -> Document:
    return Document(page_content=text)


def add_document_from_text(text: str):
    vectorstore = get_vectorstore()
    document = create_document(text)
    vectorstore.add_documents([document])


def add_document_from_import(file_metadata: dict):
    vectorstore = get_vectorstore()
    documents = load_and_process_documents(file_metadata)
    vectorstore.add_documents(documents)

