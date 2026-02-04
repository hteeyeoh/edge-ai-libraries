from .config import config
from .document import load_file_document
from .logger import logger
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
import importlib
import pandas as pd

from urllib.parse import urlparse
from langchain_vdms.vectorstores import VDMS, VDMS_Client
from langchain_core.embeddings import Embeddings
from typing import Any, List
import requests
import sys
import uuid

import base64
import json
from PIL import Image
import os
import io

vectorstore = None

# Embedding / VDMS Configuration
VDMS_HOST = os.getenv("VDMS_HOST", "localhost")
VDMS_PORT = int(os.getenv("VDMS_PORT", "5555"))
EMBEDDING_HOST = os.getenv("EMBEDDING_HOST", "localhost")
EMBEDDING_PORT = int(os.getenv("EMBEDDING_PORT", "5000"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")
ENABLE_EMBEDDING = os.environ.get("ENABLE_EMBEDDING", "false").lower() in ("true", "1", "yes")
EMBEDDING_LENGTH: int = 0

# Proxy settings
NO_PROXY_ENV = os.environ.get("no_proxy", "")  # Comma-separated domains for no-proxy
HTTP_PROXY = os.environ.get("http_proxy", "")
HTTPS_PROXY = os.environ.get("https_proxy", "")

vdms_client = VDMS_Client(host="10.223.22.126", port=55555)

# The RUN_TEST flag is used to bypass the model download and conversion steps during pytest unit testing.
# If RUN_TEST is set to "True", the model download and conversion steps are skipped.
# This flag is set in the conftest.py file before running the tests.
if os.getenv("RUN_TEST", "").lower() != "true":
    if config.MODEL_RUNTIME == "openvino":
        runtime_module = importlib.import_module("app.openvino_backend")
        runtime_instance = runtime_module.OpenVINOBackend()

    elif config.MODEL_RUNTIME == "ollama":
        runtime_module = importlib.import_module("app.ollama_backend")
        runtime_instance = runtime_module.OllamaBackend()

    else:
        raise ValueError(f"Unsupported model runtime: {config.MODEL_RUNTIME}")

    embedding, llm, reranker = runtime_instance.init_models()

    template = config.PROMPT_TEMPLATE

    prompt = ChatPromptTemplate.from_template(template)

else:
    logger.info("Bypassing to mock these functions because RUN_TEST is set to 'True' to run pytest unit test.")


def default_context(docs):
    """
    Returns a default context when the retriever is None.

    This function is used to provide a default context in scenarios where
    the retriever is not available or not provided.

    Returns:
        str: An empty string as the default context.
    """

    return ""


def get_retriever():
    """
    Creates and returns a retriever object with optional reranking capability.

    Returns:
        retriever: A retriever object, optionally wrapped with a contextual compression reranker.

    """

    embedding_api_url = "http://10.223.22.126:9777/embeddings"

    embeddings = EmbeddingAPI(
        api_url=embedding_api_url,
        model_name=EMBEDDING_MODEL_NAME
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
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.0 # >= score_threshold
        },
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
        sys_embedding = EMBEDDING_LENGTH

        if sys_embedding > 0:
            return sys_embedding

        sample_embedding = self.embed_documents(["probe_text"])
        if not sample_embedding or not isinstance(sample_embedding[0], list):
            raise ValueError("Embedding service returned invalid probe response")

        sys_embedding = len(sample_embedding[0])
        logger.debug("Embedding dimension detected: %d", sys_embedding)
        return sys_embedding


def build_chain(retriever=None):
    """
    Builds a Retrieval-Augmented Generation (RAG) chain using the provided retriever.

    Args:
        retriever: A retriever object that fetches relevant documents based on a query.

    Returns:
        A RAG chain that processes the context and question, and generates a response.
    """

    context = retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs))

    chain = (
        {
            "context": context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

#async def process_query(chain=None, query: str = ""):
async def process_query(retriever, chain=None, query: str = ""):
    """
    Processes a query using the provided chain and yields the results asynchronously.
    Args:
        chain: An optional chain object that has an `astream` method to process the query.
        query (str): The query string to be processed.
    Yields:
        str: The processed data chunks in the format "data: {chunk}\n\n".
    """

    #async for chunk in chain.astream(query):
    #    yield f"data: {chunk}\n\n"

    # 1) Retrieve documents first (so metadata is available immediately)
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

    # Send metadata once
    #yield "event: sources\n"
    #yield f"data: {json.dumps(sources, ensure_ascii=False)}\n\n"

    # 2) Build context string for the LLM
    #context = "\n\n".join(d.page_content for d in docs)

    # 3) Stream the LLM answer
    async for chunk in chain.astream(query):
        yield f"data: {chunk}\n\n"
        #yield "event: token\n"
        #yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"


    # Done marker
    # yield "event: frame\n"
    # yield f"data: {json.dumps(sources)}\n\n"
    # yield f"data: {sources}\n\n"


def create_faiss_vectordb(file_path: str = "", chunk_size=1000, chunk_overlap=200):
    """
    Creates a FAISS vector database from a document file.
    This function loads a document from the specified file path, splits it into chunks,
    creates embeddings for the chunks, and stores them in a FAISS vector database. If a
    global vectorstore already exists, it merges the new embeddings into the existing
    vectorstore.

    Args:
        file_path (str): The path to the document file. Defaults to an empty string.
        chunk_size (int): The size of each chunk in characters. Defaults to 1000.
        chunk_overlap (int): The number of overlapping characters between chunks. Defaults to 200.

    Returns:
        bool: True if the vector database was created successfully.
    """

    global vectorstore
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Load the document from the /tmp path and create embedding
    docs = load_file_document(file_path)
    splits = text_splitter.split_documents(docs)

    if not splits:
        logger.error("No text data from the document.")
        return False

    doc_embedding = FAISS.from_documents(documents=splits, embedding=embedding)
    if vectorstore == None:
        vectorstore = doc_embedding
    else:
        vectorstore.merge_from(doc_embedding)
    '''

    return True


def get_document_from_vectordb():
    """
    Retrieve document names from the vector database.
    This function accesses the global `vectorstore` object, extracts document
    metadata, and returns a list of document names.

    Returns:
        []: Return empty list if the `vectorstore` is None.
        list: A list of document names extracted from the vector database.
    """

    global vectorstore

    if vectorstore is None:
        return []

    vstore = vectorstore.docstore._dict

    docs = {vstore[key].metadata["source"].split("/")[-1] for key in vstore.keys()}

    return list(docs)


def delete_embedding_from_vectordb(document: str = "", delete_all: bool = False):
    """
    Deletes embeddings from the vector database.

    Args:
        document (str): The name of the document whose embeddings are to be deleted. If empty, no specific document is targeted.
        delete_all (bool): If True, all embeddings in the vector database will be deleted. If False, only embeddings related to the specified document will be deleted.

    Returns:
        bool: True if the deletion process completes successfully.
    """

    global vectorstore

    if vectorstore is None:
        return False

    vstore = vectorstore.docstore._dict
    data_rows = []

    for key in vstore.keys():
        doc_name = vstore[key].metadata["source"].split("/")[-1]
        content = vstore[key].page_content
        data_rows.append(
            {
                "chunk_id": key,
                "document": doc_name,
                "content": content,
            }
        )

    vectordf = pd.DataFrame(data_rows)

    if delete_all:
        # delete all the embeddings in vectorstore
        chunk_list = vectordf["chunk_id"].tolist()
    else:
        # delete the specified document embeddings in vectorstore
        chunk_list = vectordf.loc[vectordf["document"] == document]["chunk_id"].tolist()

    vectorstore.delete(chunk_list)

    return True
