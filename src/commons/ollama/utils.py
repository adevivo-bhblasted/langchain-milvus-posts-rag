from langchain_ollama import OllamaEmbeddings
from ..constants.ollama import Constants as OllamaConstants

ollama_embeddings = OllamaEmbeddings(
    model=OllamaConstants.OLLAMA_EMBEDDING_MODEL,
)

def get_embeddings_from_ollama(text: str):
    embedding = ollama_embeddings.embed_query(text)
    return embedding