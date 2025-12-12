
# /home/pipeline-server/gvapython/publisher/normalize.py
from langchain_vdms.vectorstores import VDMS, VDMS_Client
from langchain_core.embeddings import Embeddings
from typing import Any, Dict, List
from threading import Lock
import json
import logging
import os
import requests
import uuid

from gstgva import VideoFrame

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

class Normalize:
    def __init__(self):
        self.get_env_variables()

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
            collection_name="captions_collection",
            engine="FaissFlat",
            distance_strategy="IP",
            embedding_dimensions=512
        )

        # NEW: Reuse an http session for faster repeated calls to embedding service
        self._http = requests.Session()
        self._http.headers.update({'Content-Type': 'application/json'})




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

    def process(self, frame: VideoFrame):
        """
        After gvagenai:
          - Extract caption text from messages (JSON string) or tensors
          - Remove all existing messages, ROIs, and tensors
          - Add one clean caption-only JSON string message
        """

        # --- 1) Extract caption text from messages (JSON strings) ---
        caption_text = None
        metrics = {}
        for msg in frame.messages():
            try:
                if isinstance(msg, str):
                    data = json.loads(msg)

                    # Collect metrics if present
                    if "metrics" in data:
                        metrics.update(data["metrics"])

                    # Common keys: 'caption', 'text', 'result' (or nested under 'message')
                    caption_text = (
                        data.get("caption") or
                        data.get("text") or
                        data.get("result")
                    )
                    if caption_text is None and isinstance(data.get("message"), dict):
                        caption_text = (
                            data["message"].get("caption") or
                            data["message"].get("text") or
                            data["message"].get("result")
                        )
                    if caption_text:
                        break
            except Exception:
                # Ignore malformed JSON or other shapes
                pass

        # --- 2) Fallback: get caption from tensors (e.g., tensor named 'caption') ---
        if caption_text is None:
            try:
                for t in frame.tensors():
                    name = (t.name() or "").lower()
                    if name in ("caption", "text", "description"):
                        caption_text = t.label() or getattr(t, "data", {}).get("text", None)
                        if caption_text:
                            break
                    if not caption_text and t.label():
                        caption_text = t.label()
                        break
            except Exception:
                pass

        # --- 3) Remove all existing messages so detection JSON can't be serialized later ---
        try:
            for msg in list(frame.messages()):
                frame.remove_message(msg)  # removes GVAJSONMetaStr message
        except Exception:
            # If your build lacks remove_message(), see the advanced fallback below.
            pass

        # --- 4) Remove non-caption metadata (ROIs & tensors) ---
        try:
            for roi in list(frame.regions()):
                try:
                    roi.remove()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            for tensor in list(frame.tensors()):
                try:
                    tensor.remove()
                except Exception:
                    pass
        except Exception:
            pass
        # --- 5) Add a single caption-only JSON string message ---
        info = frame.video_info()

        # call embedding service to get embedding
        payload = {
            "input": {
                "type": "text",
                "text":caption_text,
            },
            "model": self.embedding_model,
            "encoding_format": "float"
        }
        response = self._http.post(self.embedding_endpoint,
                                   json=payload,
                                   timeout=(1.0,2.0)
                                   )  # (connect timeout, read timeout)
        response.raise_for_status()
        data = response.json()
        emb = data.get("embedding")


        if emb is None:
            raise ValueError("Missing 'embeddings' in data")

        # If emb is a numpy array, convert:
        # if isinstance(emb, np.ndarray):
        #         emb = emb.tolist()

        # Ensure it's a flat list of numbers (single vector)
        if not isinstance(emb, (list, tuple)) or not emb:
            raise TypeError(f"Embedding must be a non-empty list/tuple, got {type(emb)}")

        vector = [float(x) for x in emb]

        meta = {
            "source_model": self.embedding_model,
            "type": "text"
        }

        # Store caption and embedding in VDMS
        ids = str(uuid.uuid4())
        self.vdms_store.add_from(
            texts=[caption_text],
            metadatas=[meta],
            embeddings=[vector],
            ids=[ids]
        )

        payload = {
            "type": "caption",
            "text": caption_text,
            "resolution": {"width": info.width, "height": info.height},
            "metrics": metrics
        }
        frame.add_message(json.dumps(payload))  # one string argument


        return True
