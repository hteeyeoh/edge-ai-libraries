from .config import config
from .logger import logger

import os
import json
import requests
from urllib.parse import urlparse
from langchain_vdms.vectorstores import VDMS, VDMS_Client
from langchain_core.embeddings import Embeddings
from typing import Any, List

# Proxy ENV
NO_PROXY_ENV = os.environ.get("no_proxy", "")  # Comma-separated domains for no-proxy
HTTP_PROXY = os.environ.get("http_proxy", "")
HTTPS_PROXY = os.environ.get("https_proxy", "")

if config.USE_VDMS:
    vdms_client = VDMS_Client(host=config.VDMS_HOST, port=config.VDMS_PORT)


class EmbeddingAPI(Embeddings):
    """Lightweight embedding client that forwards text requests to the serving API."""

    def __init__(self, api_url: str, model_name: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name

    def _post_embeddings(self, payload: dict) -> List[List[float]]:
        """Execute a POST request to the embedding service."""
        proxies = (
            None
            if should_use_no_proxy(self.api_url)
            else {"http": HTTP_PROXY, "https": HTTPS_PROXY}
        )

        try:
            response = requests.post(self.api_url, json=payload, proxies=proxies)
            logger.debug("Embedding service response status code: %s", response.status_code)
            response.raise_for_status()
            embeddings = response.json()["embedding"]
            if not isinstance(embeddings, list):
                raise ValueError("Embedding service returned unexpected payload")
            if embeddings and isinstance(embeddings[0], (int, float)):
                embeddings = [embeddings]
            return embeddings
        except requests.RequestException as exc:
            logger.debug("Failed to call embedding service: %s", exc)
            raise Exception("Error creating embedding") from exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.debug("Embedding batch of %d documents", len(texts))
        payload = {
            "model": self.model_name,
            "input": {"type": "text", "text": texts},
            "encoding_format": "float",
        }
        return self._post_embeddings(payload)

    def embed_query(self, text: str) -> List[float]:
        logger.debug("Embedding single query")
        payload = {
            "model": self.model_name,
            "input": {"type": "text", "text": text},
            "encoding_format": "float",
        }
        embeddings = self._post_embeddings(payload)
        return embeddings[0]

    def get_embedding_length(self) -> int:
        logger.debug(
            "Retrieving embedding dimensionality for model %s via API probe", self.model_name
        )
        sys_embedding = config._VDMS_EMBEDDING_LENGTH

        if sys_embedding > 0:
            return sys_embedding

        sample_embedding = self.embed_documents(["probe_text"])
        if not sample_embedding or not isinstance(sample_embedding[0], list):
            raise ValueError("Embedding service returned invalid probe response")

        sys_embedding = len(sample_embedding[0])
        logger.debug("Embedding dimension detected: %d", sys_embedding)
        return sys_embedding


def get_vdms_retriever():
    """
    Initialize VDMS retriever and return as a LangChain retriever object
    """

    embedding_api_url = f"http://{config.VDMS_EMBEDDING_HOST}:{config.VDMS_EMBEDDING_HOST_PORT}/embeddings"
    print(f"embedding_URL: {embedding_api_url}")

    embeddings = EmbeddingAPI(
        api_url=embedding_api_url,
        model_name=config.VDMS_EMBEDDING_MODEL,
    )

    vector_dimensions = embeddings.get_embedding_length()

    vdms_store = VDMS(
        client=vdms_client,
        embedding=embeddings,
        collection_name="captions_collection",
        engine="FaissFlat",
        distance_strategy="IP",
        embedding_dimensions=vector_dimensions
    )

    retriever = vdms_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    return retriever


def should_use_no_proxy(url: str) -> bool:
    no_proxy = NO_PROXY_ENV
    hostname = urlparse(url).hostname
    logger.debug(
        f"Checking no_proxy for hostname: {hostname} against no_proxy domains: {no_proxy}"
    )
    if hostname:
        for domain in no_proxy.split(","):
            domain = domain.strip()
            if not domain:
                continue
            if hostname.endswith(domain):
                logger.debug(f"Hostname {hostname} matches no_proxy domain {domain}")
                return True
    return False


async def process_vdms_query(chain=None, query: str = "", retriever=None):
    # Retrieve documents first (so metadata is available immediately)
    # retriever is a Runnable in LangChain, so `.ainvoke` works in async contexts
    docs = await retriever.ainvoke(query)

    # Example of sources data: [{'metadata': {'frame_data': 'base64_encoded', 'frame_format': 'BGRA', 'frame_height': 1080, 'frame_id': 11, 'frame_width': 1920}, 'preview': 'A white Nissan Leaf car is parked in a parking garage.'}, {'metadata': {'frame_data': 'base64_encoded', 'frame_format': 'BGRA', 'frame_height': 1080, 'frame_id': 10, 'frame_width': 1920}, 'preview': 'A white Nissan Leaf car is parked in a parking garage with its tail lights on, surrounded by marked spaces.'}, {'metadata': {'frame_data': 'base64_encoded', 'frame_format': 'BGRA', 'frame_height': 1080, 'frame_id': 4, 'frame_width': 1920}, 'preview': 'A white Nissan Leaf car is parked in a parking garage with its tail lights on, surrounded by marked spaces.'}]
    sources = [
       {
           "metadata": d.metadata,
            # optional: include a preview/snippet for UX
           "preview": d.page_content[:200],
       }
       for d in docs
    ]

    async for chunk in chain.astream(query):
        yield f"data: {chunk}\n\n"

    # Done marker
    yield "event: frame\n"
    yield f"data: {json.dumps(sources)}\n\n"
