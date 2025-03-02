from dotenv import load_dotenv
from pymilvus import (
    CollectionSchema, 
    FieldSchema, 
    DataType, 
)
from .milvus_utils.connection import milvus_client
from .commons.constants.milvus import Constants as MilvusConstants

load_dotenv()


if milvus_client.has_collection(MilvusConstants.COLLECTION_NAME):
    print(f"Collection '{MilvusConstants.COLLECTION_NAME}' exists.")
    milvus_client.drop_collection(MilvusConstants.COLLECTION_NAME)

print(f"Creating collection '{MilvusConstants.COLLECTION_NAME}'")
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512, nullable=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024, description="Embedding vector")
    ],
    description="Collection for storing text documents with embeddings",
    enable_dynamic_field=True  # Allows dynamic fields in the collection
)

collection = milvus_client.create_collection(
    collection_name=MilvusConstants.COLLECTION_NAME,
    schema=schema,
    shards_num=2
)
print(f"Collection '{MilvusConstants.COLLECTION_NAME}' created successfully!")

