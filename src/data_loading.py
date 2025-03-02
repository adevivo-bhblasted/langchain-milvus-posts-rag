import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_milvus import Milvus
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, MilvusClient

# ---- Configuration ---- #
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "demo_collection"
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "mxbai-embed-large:latest"

# ---- Connect to Milvus ---- #
try:
    client = MilvusClient(uri=MILVUS_URI)
    print(f"✅ Connected to Milvus at {MILVUS_URI}")
except Exception as e:
    print(f"❌ Milvus connection failed: {e}")
    exit()

# ---- Check & Create Collection ---- #
if client.has_collection(COLLECTION_NAME):
    print(f"⚠️ Collection '{COLLECTION_NAME}' already exists. Dropping and recreating it...")
    client.drop_collection(COLLECTION_NAME)

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512, nullable=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024, description="Embedding vector")
    ],
    description="Collection for storing text documents with embeddings",
    enable_dynamic_field=True
)

collection = client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    shards_num=2
)

print(f"✅ Collection '{COLLECTION_NAME}' successfully created!")

# ---- Text Cleaning Function ---- #
def clean_text(text):
    """Cleans and normalizes text."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters except common punctuation
    return text.strip()

# ---- Load and Process Documents ---- #
def load_and_process_documents():
    """Loads and processes documents from different sources."""
    pdf_loader = PyPDFLoader("sample.pdf")  # Replace with your PDF file
    csv_loader = CSVLoader(file_path="sample.csv")  # Replace with your CSV file

    pdf_docs = pdf_loader.load()
    csv_docs = csv_loader.load()

    # Clean text content
    for doc in pdf_docs + csv_docs:
        doc.page_content = clean_text(doc.page_content)

    all_docs = pdf_docs + csv_docs
    if not all_docs:
        print("⚠️ No documents loaded! Exiting...")
        exit()

    print(f"✅ Loaded {len(all_docs)} documents.")
    return all_docs

all_documents = load_and_process_documents()

# ---- Initialize LLM & Embeddings ---- #
llm = ChatOllama(model=LLM_MODEL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# ---- Split Documents into Chunks ---- #
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)

splitted_documents = [
    chunk for doc in all_documents for chunk in text_splitter.split_documents([doc])
]

print(f"✅ Total document chunks after splitting: {len(splitted_documents)}")

# ---- Validate Embeddings ---- #
test_embedding = embeddings.embed_query("Test sentence")
if len(test_embedding) != 1024:
    raise ValueError(f"❌ Embedding model returned {len(test_embedding)} dimensions instead of 1024!")

print(f"✅ Embedding model correctly produces {len(test_embedding)}-dimensional vectors.")

# ---- Insert Documents into Milvus ---- #
vectorstore = Milvus.from_documents(
    documents=splitted_documents,
    embedding=embeddings,
    connection_args={"uri": MILVUS_URI},
    collection_name=COLLECTION_NAME,
)

print("✅ Documents successfully inserted into Milvus!")