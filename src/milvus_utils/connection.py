from pymilvus import (
    MilvusClient
)
from commons.constants.milvus import Constants as MilvusConstants



def load_milvus_connection():
    # Define Milvus connection URI
    milvus_uri = f"http://{MilvusConstants.HOST_NAME}:{MilvusConstants.PORT}"
    # Attempt to connect to Milvus
    try:
        client = MilvusClient(uri=milvus_uri)
        print(f"✅ Connected to Milvus at {milvus_uri}")
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        exit()
    
    return client


milvus_client = load_milvus_connection()