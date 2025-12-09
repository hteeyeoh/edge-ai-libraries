from langchain_vdms.vectorstores import VDMS, VDMS_Client
from langchain_core.embeddings import Embeddings
from typing import Any, List
import requests
import sys
import uuid

class DummyEmbedding(Embeddings):
    """
    Minimal dummy embedding class that satisfies VDMS requirements.
    We won't actually use these methods since we use add_from() directly.
    """

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Won't be called since we use add_from() directly."""
        raise NotImplementedError("Use add_from() method instead")

    def embed_query(self, text: str) -> List[float]:
        """Won't be called since we use add_from() directly."""
        raise NotImplementedError("Use add_from() method instead")

# Check for user input
if len(sys.argv) < 2:
    print("Usage: python3 pipeline.py \"<your_query>\"")
    sys.exit(1)

# Step 1: Call the embeddings API
embedding_api_url = "http://localhost:9777/embeddings"
text_contents = sys.argv[1]  # Get the text from command line
headers = {"Content-Type": "application/json"}
payload = {
    "input": {
        "type": "text",
        "text": text_contents
    },
    "model": "CLIP/clip-vit-b-32",
    "encoding_format": "float"
}


# Send request to get embedding
response = requests.post(embedding_api_url, headers=headers, json=payload)
response.raise_for_status()
embedding_data = response.json()
embeddings = embedding_data.get("embedding")


# Step 1: Connect to VDMS
vdms_client = VDMS_Client(host="localhost", port=55555)

dummy_embedding = DummyEmbedding()

vdms_store = VDMS(
    client=vdms_client,
    embedding=dummy_embedding,
    collection_name="caption_collection",
    engine="FaissFlat",
    distance_strategy="IP",
    embedding_dimensions=512
    )


results = vdms_store.similarity_search_by_vector(embeddings, k=3)


print("\nSearch Results:")
for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(f"ID: {doc.id}")
    print(f"Metadata:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    print("Content:")
    print(f"  {doc.page_content}")
