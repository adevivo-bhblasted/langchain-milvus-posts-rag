from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from milvus_client.connection import milvus_client
from commons.constants.milvus import Constants as MilvusConstants
from commons.constants.ollama import Constants as OllamaConstants


def query_posts(query_str: str) -> str:
    # ---- Check for Collection ---- #
    if MilvusConstants.COLLECTION_NAME not in milvus_client.list_collections():
        print(f"⚠️ Collection '{MilvusConstants.COLLECTION_NAME}' not found in Milvus!")
        exit()

    # ---- Initialize LLM & Embeddings ---- #
    llm = ChatOllama(model=OllamaConstants.OLLAMA_CHAT_MODEL)
    embeddings = OllamaEmbeddings(model=OllamaConstants.OLLAMA_EMBEDDING_MODEL)


    # ---- Initialize Vector Store ---- #
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"host": MilvusConstants.HOST_NAME, "port": MilvusConstants.PORT},
        collection_name=MilvusConstants.COLLECTION_NAME,
    )

    # ---- Define Prompt Template ---- #
    PROMPT_TEMPLATE = """Human: You are an AI assistant providing fact-based answers using statistical information where possible.
    Use the following context to answer the question enclosed in <question> tags.
    If you don't know the answer, say that you don't know; don't make up an answer.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    Provide specific responses with statistics or numbers when available.

    Assistant:"""

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    # ---- Convert Vector Store to Retriever ---- #
    retriever = vectorstore.as_retriever()

    # ---- Format Retrieved Documents ---- #
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ---- Define the RAG Chain ---- #
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ---- Run Query ---- #
    response = rag_chain.invoke(query_str)

    # ---- Return Response ---- #
    return response

