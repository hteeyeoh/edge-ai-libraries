from langchain_vdms.vectorstores import VDMS, VDMS_Client
from langchain_core.embeddings import Embeddings
from typing import Any, List
import requests
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


# Step 1: Call the embeddings API
embedding_api_url = "http://localhost:9777/embeddings"
#text_contents = "A white box with a green surface is in the picture."
text_contents = "What the persons wearing?"
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
#print(f"embeddings: {embeddings}")


# Step 1: Connect to VDMS
#vdms_client = VDMS_Client(host="vdms-vector-db", port=55555)
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



#metadatas = [
#    {"source": "clip-vit-b-32", "id": "doc001"},
#    {"source": "clip-vit-b-32", "id": "doc002"},
#    {"source": "clip-vit-b-32", "id": "doc003"}
#]

#ids = [str(uuid.uuid4()) for _ in range(len(text_contents))]

#vdms_store.add_from(
#    texts=[text_contents],
#    metadatas=metadatas,
#    embeddings=[embeddings],
#    ids=ids
#)

#print("Inserted text with external embedding into VDMS.")

results = vdms_store.similarity_search_by_vector(embeddings, k=3)

print("Search Results:", results)
