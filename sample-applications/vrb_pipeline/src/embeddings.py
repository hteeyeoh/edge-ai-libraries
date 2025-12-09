# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from langchain_vdms.vectorstores import VDMS, VDMS_Client
from langchain_core.embeddings import Embeddings
from typing import Any, Dict, List
from threading import Lock
import json
import logging
import os
import requests
import uuid

logger = logging.getLogger('EMBEDDING_PUBLISHER')
logger.setLevel(logging.DEBUG)

class DummyEmbeddings(Embeddings):
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


class EmbeddingsPublisher:
    def __init__(self,
                 metadata_path: str = "/tmp/frames_metadata.json",
                 write_mode: str = "incremental"
                 ):

        """
        Args:
            metadata_path: Path to the original metadata.json (list of objects).
            write_mode: 'incremental' to write after each frame; 'final' to write in destructor.
        """

        try:
            logger.info("Initializing EmbeddingsPublisher via gvapython extension...")

            self.get_env_variables()

            self._lock = Lock()


            # Initialize embedding resources
            dummy_embedding = DummyEmbeddings()

            self.embedding_endpoint = f"http://{self.embedding_host}:{self.embedding_port}/embeddings"

            # Initialize VDMS client connection

            self.vdms_client = VDMS_Client(
                host = self.vdms_host,
                port = self.vdms_port
            )

            self.vdms_store = VDMS(
                client=self.vdms_client,
                embedding=dummy_embedding,
                collection_name="caption_collection",
                engine="FaissFlat",
                distance_strategy="IP",
                embedding_dimensions=512
            )

            self._metadata_path = metadata_path
            self._write_mode = write_mode  # 'incremental' or 'final'

            # NEW: Keep an list of caption + embedding metadata records
            self.embedding_records: List[Dict[str, Any]] = []

            # NEW: Reuse an http session for faster repeated calls to embedding service
            self._http = requests.Session()
            self._http.headers.update({'Content-Type': 'application/json'})

            if not os.path.exists(self._metadata_path):
                raise FileNotFoundError(f"metadata.json not found at {self._metadata_path}")

            # open existing metadata.json
            print(f"Loading metadata from {self._metadata_path}...")
            with open(self._metadata_path, 'r', encoding='utf-8') as f:
                self.metadata: List[Dict[str, Any]] = [json.loads(l) for l in f if l.strip()]

            print(self.metadata)


            if not isinstance(self.metadata, list):
                raise ValueError("metadata.json must be a JSON array that should contain a list of objects.")

            # 1-based index to match multifilesrc start-index=1
            self._idx = 1

            logger.info("EmbeddingsPublisher initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingsPublisher: {str(e)}")
            raise

    def __del__(self):
        """Destructor to clean up resources."""
        if self._write_mode == "final":
            print("Writing final metadata to file...")
            self._write_inplace()

        # Flush embeddings to VDMS
        self.flush_embeddings_to_vdms()

        print("Pipeline finished. Destroying resources...")

    def get_env_variables(self):
        try:
            print("Getting environment variables for EmbeddingsPublisher...")
            self.vdms_host: str = os.getenv("VDMS_HOST", "localhost")
            self.vdms_port: int = int(os.getenv("VDMS_PORT", "55555"))
            self.embedding_host: str = os.getenv("EMBEDDING_HOST", "localhost")
            self.embedding_port: int = int(os.getenv("EMBEDDING_PORT", "8000"))
            self.embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME", "")

        except ValueError:
            logger.error("Port value should be an integer.")
            raise Exception("Port value should be an integer.")

    def _write_inplace(self):
        """Writes the updated metadata list back to metadata.json."""
        try:
            with self._lock:
                with open(self._metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=4)

        except Exception as e:
            logger.error(f"Failed to write updated metadata: {str(e)}")

    def process(self, frame):
        try:
            caption = None

            with frame.data() as image:
                video_info = frame.video_info()
                metadata = self.get_gva_metadata(frame.messages())
                caption = metadata.get("result")

                index = self._idx - 1
                if 0 <= index < len(self.metadata):
                    if caption is not None:
                        self.metadata[index]["caption"] = caption
                    else:
                        self.metadata[index].setdefault("caption", None)

                    # New: call embedding service to get embedding and store it in list
                    if caption:
                        try:
                            payload = {
                                "input": {
                                    "type": "text",
                                    "text": caption
                                },
                                "model": self.embedding_model,
                                "encoding_format": "float"
                            }
                            response = self._http.post(self.embedding_endpoint,
                                                       json=payload,
                                                       timeout=(1.0,2.0)
                                                       ) # (connect timeout, read timeout)
                            response.raise_for_status()
                            data =  response.json()

                            # Get embeddings from the data
                            embeddings = None
                            if isinstance(data, dict):
                                embeddings = data.get("embedding")

                            # Form a metadata with caption and embedding to store in vdms
                            if embeddings is not None:
                                frame_index_value = self.metadata[index].get("frame_index")
                                record = {
                                    "text": caption,
                                    "embeddings": embeddings,
                                    "metadata": {"id": f"frame_{frame_index_value}",
                                                 "source_model": self.embedding_model
                                                }
                                }
                                self.embedding_records.append(record)

                            else:
                                logger.warning(f"No embeddings found in response for caption: {caption}")

                        except Exception as e:
                            logger.error("Failed to get embeddings for caption")

                    if self._write_mode == "incremental":
                        self._write_inplace()

                self._idx += 1

                return True

        except Exception as e:
            logger.error(f"Error processing frame in EmbeddingsPublisher: {str(e)}")
            return True


    def flush_embeddings_to_vdms(self) -> None:
        """Flushes accumulated embeddings to VDMS in batch."""
        try:
            if not self.embedding_records:
                logger.info("No embeddings to flush to VDMS.")
                return

            if self.vdms_store is None:
                logger.error("VDMS store is not initialized.")
                return

            text_contents: List[str] = []
            embeddings: List[List[float]] = []
            metadatas: List[Dict[str, Any]] = []

            # Validate and collect
            for rec in self.embedding_records:
                txt = rec.get("text")
                emb = rec.get("embeddings")
                meta = rec.get("metadata")

                if txt is None or emb is None or meta is None:
                    logger.warning(f"Skipping invalid record (missing keys): {rec}")
                    continue

                # Ensure embeddings is a flat list/tuple of numbers
                if not isinstance(emb, (list, tuple)):
                    logger.warning(f"Embedding must be list/tuple, got {type(emb)}; skipping")
                    continue
                # Optionally enforce element type
                try:
                    emb_list = [float(x) for x in emb]

                except Exception:
                    logger.warning("Embedding contains non-numeric values; skipping record.")
                    continue

                # Collect
                text_contents.append(txt)
                embeddings.append(emb_list)
                metadatas.append(meta)

            if not text_contents:
                logger.info("No valid records after validation; nothing to upload.")
                return

            # Generate unique IDs (one per item)
            ids = [str(uuid.uuid4()) for _ in range(len(text_contents))]

            logger.info(f"Uploading {len(text_contents)} records to VDMS...")
            self.vdms_store.add_from(
                texts=text_contents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            logger.info("Upload to VDMS completed.")

        except Exception as e:
            logger.error(f"Failed to flush embeddings to VDMS: {e}")


    def get_gva_metadata(self, messages:list) -> dict:
        """Takes a list of frame meta messages, loads them as a JSON and
        updates the metadata dict with the loaded JSON.
        """

        metadata: dict = {}
        for message in messages:
            message_json = json.loads(message)
            metadata.update(message_json)

        return metadata