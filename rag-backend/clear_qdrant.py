from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()
try:
    print(f"Connecting to Qdrant at {os.getenv('QDRANT_ENDPOINT')}...")
    client = QdrantClient(
        url=os.getenv('QDRANT_ENDPOINT'), 
        api_key=os.getenv('QDRANT_API_KEY'),
        timeout=60.0
    )

    col_name = os.getenv('QDRANT_COLLECTION_NAME', 'document_chunks')
    print(f"Deleting collection {col_name}...")
    client.delete_collection(collection_name=col_name)
    print(f"Collection '{col_name}' deleted.")
except Exception as e:
    print(f"Failed: {e}")
