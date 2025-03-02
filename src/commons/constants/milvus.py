import os

class Constants:
    HOST_NAME = os.getenv("MILVUS_HOST", "localhost")
    PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "posts")

