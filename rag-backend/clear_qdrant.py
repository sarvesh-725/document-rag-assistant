from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

from app.services.rag_engine import QDRANT_COLLECTION_NAME

load_dotenv()
try:
    print(f"Connecting to Qdrant at {os.getenv('QDRANT_ENDPOINT')}...")
    client = QdrantClient(
        url=os.getenv('QDRANT_ENDPOINT'), 
        api_key=os.getenv('QDRANT_API_KEY'),
        timeout=60.0
    )

    print(f"Deleting collection {QDRANT_COLLECTION_NAME}...")
    client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
    print(f"Collection '{QDRANT_COLLECTION_NAME}' deleted.")
except Exception as e:
    print(f"Failed: {e}")
