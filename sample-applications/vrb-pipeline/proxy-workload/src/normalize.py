
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
        After gvagenai + gvametaconvert:
          - Extract caption text (messages/tensors)
          - Call embedding service; store caption+embedding in VDMS
          - Merge caption+metrics into detection JSON
          - Remove the ORIGINAL detection JSON string and set merged as FIRST
          - Do NOT remove ROIs/tensors
        """
        # ------------------------------------------------------------
        # 0) Collect messages and keep RAW strings so we can remove by value
        # ------------------------------------------------------------
        det_obj = None
        cap_obj = None
        det_raw_msg = None
        cap_raw_msg = None

        for msg in list(frame.messages()):
            if not isinstance(msg, str):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            if "objects" in data and "resolution" in data:
                det_obj = data
                det_raw_msg = msg               # keep ORIGINAL string to remove by value
            elif ("result" in data and isinstance(data["result"], str)) or "metrics" in data:
                cap_obj = data
                cap_raw_msg = msg               # may remove if needed

        # ------------------------------------------------------------
        # 1) Extract caption + metrics
        # ------------------------------------------------------------
        caption_text = None
        metrics = {}
        cap_ts = None
        cap_ts_seconds = None

        if isinstance(cap_obj, dict):
            caption_text = (
                cap_obj.get("caption")
                or cap_obj.get("text")
                or cap_obj.get("result")
                or (
                    cap_obj.get("message", {}).get("result")
                    if isinstance(cap_obj.get("message"), dict) else None
                )
            )
            if isinstance(cap_obj.get("metrics"), dict):
                metrics = cap_obj["metrics"]
            cap_ts = cap_obj.get("timestamp")
            cap_ts_seconds = cap_obj.get("timestamp_seconds")

        # ------------------------------------------------------------
        # 2) Fallback: caption from tensors
        # ------------------------------------------------------------
        if not caption_text:
            try:
                for t in frame.tensors():
                    name = (t.name() or "").lower()
                    if name in ("caption", "text", "description"):
                        try:
                            caption_text = t.label()
                        except Exception:
                            caption_text = getattr(getattr(t, "data", {}), "get", lambda *_: None)("text", None)
                        if caption_text:
                            break
                    if not caption_text:
                        try:
                            if t.label():
                                caption_text = t.label()
                                break
                        except Exception:
                            pass
            except Exception:
                pass

        # ------------------------------------------------------------
        # 3) Embedding + VDMS store (only if caption available)
        # ------------------------------------------------------------
        caption_id = None
        if caption_text:
            try:
                payload = {
                    "input": {"type": "text", "text": caption_text},
                    "model": self.embedding_model,
                    "encoding_format": "float",
                }
                resp = self._http.post(self.embedding_endpoint, json=payload, timeout=(1.0, 2.0))
                resp.raise_for_status()
                rj = resp.json()
                emb = rj.get("embedding")

                if emb is None:
                    raise ValueError("Missing 'embedding' in response")

                if not isinstance(emb, (list, tuple)) or not emb:
                    raise TypeError(f"Embedding must be a non-empty list/tuple, got {type(emb)}")

                vector = [float(x) for x in emb]
                meta = {"source_model": self.embedding_model, "type": "text"}
                caption_id = str(uuid.uuid4())

                self.vdms_store.add_from(
                    texts=[caption_text],
                    metadatas=[meta],
                    embeddings=[vector],
                    ids=[caption_id],
                )
            except Exception:
                # Swallow embedding errors to keep pipeline flowing
                caption_id = None

        # ------------------------------------------------------------
        # 4) Build merged JSON starting from detection JSON (if present)
        # ------------------------------------------------------------
        merged = det_obj.copy() if isinstance(det_obj, dict) else {}

        # If no detection JSON yet, at least include resolution
        if not merged:
            try:
                info = frame.video_info()
                merged["resolution"] = {"width": info.width, "height": info.height}
            except Exception:
                pass

        if caption_text:
            merged["caption"] = caption_text

        genai_block = {}
        if metrics:
            genai_block["metrics"] = metrics
        if cap_ts is not None:
            genai_block["timestamp"] = cap_ts
        if cap_ts_seconds is not None:
            genai_block["timestamp_seconds"] = cap_ts_seconds
        if genai_block:
            merged["genai"] = genai_block

        merged_str = json.dumps(merged, separators=(",", ":"))

        # ------------------------------------------------------------
        # 5) Replace messages so merged JSON is FIRST
        #    - remove the ORIGINAL detection message by value
        #    - optionally remove caption message
        #    - clear any remaining and set merged
        # ------------------------------------------------------------
        try:
            # Remove original detection JSON string explicitly (by value)
            if det_raw_msg is not None:
                frame.remove_message(det_raw_msg)

            # Optional: remove original caption message to avoid duplicates
            if cap_raw_msg is not None:
                try:
                    frame.remove_message(cap_raw_msg)
                except Exception:
                    pass

            # Remove any other messages to guarantee ours is first
            # (some builds publish only the first message)
            while frame.messages():
                try:
                    frame.remove_message(0)
                except TypeError:
                    frame.remove_message(frame.messages()[0])
        except Exception:
            # If removal is limited, we still add ours; many builds publish the first message
            pass

        frame.add_message(merged_str)
        return True





    # def process(self, frame: VideoFrame):
    #     """
    #     After gvagenai:
    #       - Extract caption text from messages (JSON string) or tensors
    #       - Remove all existing messages, ROIs, and tensors
    #       - Add one clean caption-only JSON string message
    #     """

    #     # --- 1) Extract caption text from messages (JSON strings) ---
    #     caption_text = None
    #     metrics = {}
    #     for msg in frame.messages():
    #         try:
    #             if isinstance(msg, str):
    #                 data = json.loads(msg)

    #                 # Collect metrics if present
    #                 if "metrics" in data:
    #                     metrics.update(data["metrics"])

    #                 # Common keys: 'caption', 'text', 'result' (or nested under 'message')
    #                 caption_text = (
    #                     data.get("caption") or
    #                     data.get("text") or
    #                     data.get("result")
    #                 )
    #                 if caption_text is None and isinstance(data.get("message"), dict):
    #                     caption_text = (
    #                         data["message"].get("caption") or
    #                         data["message"].get("text") or
    #                         data["message"].get("result")
    #                     )
    #                 if caption_text:
    #                     break
    #         except Exception:
    #             # Ignore malformed JSON or other shapes
    #             pass

    #     # --- 2) Fallback: get caption from tensors (e.g., tensor named 'caption') ---
    #     if caption_text is None:
    #         try:
    #             for t in frame.tensors():
    #                 name = (t.name() or "").lower()
    #                 if name in ("caption", "text", "description"):
    #                     caption_text = t.label() or getattr(t, "data", {}).get("text", None)
    #                     if caption_text:
    #                         break
    #                 if not caption_text and t.label():
    #                     caption_text = t.label()
    #                     break
    #         except Exception:
    #             pass

    #     # --- 3) Remove all existing messages so detection JSON can't be serialized later ---
    #     try:
    #         for msg in list(frame.messages()):
    #             frame.remove_message(msg)  # removes GVAJSONMetaStr message
    #     except Exception:
    #         # If your build lacks remove_message(), see the advanced fallback below.
    #         pass

    #     # --- 4) Remove non-caption metadata (ROIs & tensors) ---
    #     try:
    #         for roi in list(frame.regions()):
    #             try:
    #                 roi.remove()
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass

    #     try:
    #         for tensor in list(frame.tensors()):
    #             try:
    #                 tensor.remove()
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass
    #     # --- 5) Add a single caption-only JSON string message ---
    #     info = frame.video_info()

    #     # call embedding service to get embedding
    #     payload = {
    #         "input": {
    #             "type": "text",
    #             "text":caption_text,
    #         },
    #         "model": self.embedding_model,
    #         "encoding_format": "float"
    #     }
    #     response = self._http.post(self.embedding_endpoint,
    #                                json=payload,
    #                                timeout=(1.0,2.0)
    #                                )  # (connect timeout, read timeout)
    #     response.raise_for_status()
    #     data = response.json()
    #     emb = data.get("embedding")


    #     if emb is None:
    #         raise ValueError("Missing 'embeddings' in data")

    #     # If emb is a numpy array, convert:
    #     # if isinstance(emb, np.ndarray):
    #     #         emb = emb.tolist()

    #     # Ensure it's a flat list of numbers (single vector)
    #     if not isinstance(emb, (list, tuple)) or not emb:
    #         raise TypeError(f"Embedding must be a non-empty list/tuple, got {type(emb)}")

    #     vector = [float(x) for x in emb]

    #     meta = {
    #         "source_model": self.embedding_model,
    #         "type": "text"
    #     }

    #     # Store caption and embedding in VDMS
    #     ids = str(uuid.uuid4())
    #     self.vdms_store.add_from(
    #         texts=[caption_text],
    #         metadatas=[meta],
    #         embeddings=[vector],
    #         ids=[ids]
    #     )

    #     payload = {
    #         "type": "caption",
    #         "text": caption_text,
    #         "resolution": {"width": info.width, "height": info.height},
    #         "metrics": metrics
    #     }
    #     frame.add_message(json.dumps(payload))  # one string argument


    #     return True
