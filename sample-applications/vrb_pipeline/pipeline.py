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
text_content = """The image depicts an animated character sitting at a desk in what appears to be a high-tech or control room environment.
        The character is wearing glasses and a lanyard with a badge, suggesting they might be an employee or technician.
        They are using a computer keyboard and have a cup of coffee on the desk, indicating they are working or studying for an extended period.

        In the background, there are multiple monitors displaying complex code or data, reinforcing the idea that the setting is related to technology or programming."""
headers = {"Content-Type": "application/json"}
payload = {
    "input": {
        "type": "text",
        "text": text_content
        #"text": """The image depicts an animated character sitting at a desk in what appears to be a high-tech or control room environment.
        #The character is wearing glasses and a lanyard with a badge, suggesting they might be an employee or technician.
        #They are using a computer keyboard and have a cup of coffee on the desk, indicating they are working or studying for an extended period.

       # In the background, there are multiple monitors displaying complex code or data, reinforcing the idea that the setting is related to technology or programming."""
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
#vdms_client = VDMS_Client(host="vdms-vector-db", port=55555)
vdms_client = VDMS_Client(host="localhost", port=55555)

dummy_embedding = DummyEmbedding()

vdms_store = VDMS(
    client=vdms_client,
    embedding=dummy_embedding,
    collection_name="text_collection",
    engine="FaissFlat",
    distance_strategy="IP",
    embedding_dimensions=512
    )


vdms_store.add_from(
    texts=[text_content],
    metadatas=[{"source": "clip-vit-b-32", "id": "doc001"}],
    embeddings=[embeddings],
    ids=[str(uuid.uuid4())]
)

print("Inserted text with external embedding into VDMS.")

results = vdms_store.similarity_search_by_vector(embeddings, k=3)

print("Search Results:", results)
