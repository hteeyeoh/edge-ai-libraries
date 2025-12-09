# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import time
import datetime
import json
import logging
import os
import numpy as np
from threading import Lock
from typing import Set
from PIL import Image


logger = logging.getLogger('FRAME_SELECTOR')
logger.setLevel(logging.DEBUG)

class FrameSelector:
    def __init__(self, *args, **kwargs):
        try:
            logger.info("Initializing FrameSelector via gvapython extension...")

            self.get_env_variables()
            self.interested: list = kwargs.get("interested", None)

            self.output_dir = "/tmp/best_frames"
            self.frame_id = 1
            self._lock = Lock()
            os.makedirs(self.output_dir, exist_ok=True)

            # Whether to save full frame instead of ROI crop
            self.save_full_frame = True

            # State per object_id
            self.best_scores = {}     # object_id -> float
            self.best_frames = {}     # object_id -> numpy array (crop or full)
            self.best_meta   = {}     # object_id -> dict {rect, confidence, label, ts}

            # Track currently visible IDs from previous iteration
            self.prev_visible_ids = set()

            logger.info("FrameSelector initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize FrameSelector: {str(e)}")
            raise

    def __del__(self):
        # On pipeline stop, flush any remaining best crops to disk
        print("deleting resources")
        # try:
        #     for oid in list(self.best_frames.keys()):
        #         self._save_best_and_cleanup(oid)
        # except Exception:
        #     pass

    def get_env_variables(self):
        try:
            print("Getting environment variables for FrameSelector...")

        except ValueError:
            logger.error("Port value should be an integer.")
            raise Exception("Port value should be an integer.")

    def process(self, frame):

        current_visible_ids = set()

        with frame.data() as np_frame:
            video_info = frame.video_info()
            metadata = self.get_gva_metadata(frame.messages())

            fmt = video_info.to_caps().get_structure(0).get_value('format')

            objects = metadata.get("objects", [])

            for obj in objects:
                obj_id = obj.get("id")
                current_visible_ids.add(obj_id)

                obj_label = obj.get("detection", {}).get("label", "")

                confidence = obj.get("detection", {}).get("confidence", 0)
                w = obj.get("w", 0)
                h = obj.get("h", 0)
                score = confidence * (w * h)

                current_best = self.best_frames.get(obj_id)
                if current_best is None or score > current_best["score"]:
                    # Update best frame for this object
                    self.best_frames[obj_id] = {
                        "score": score,
                        "metadata": {
                            "object_id": obj_id,
                            "width": w,
                            "height": h,
                            "confidence": confidence,
                            "bbox": obj.get("detection", {}).get("bounding_box", {}),
                            "label": obj.get("detection", {}).get("label", ""),
                            "img_format": fmt
                        },
                        "image": np_frame.copy()  # Store full frame
                    }

            # After processing this frame: any object that disappeared?
            disappeared = self.prev_visible_ids - current_visible_ids

            # Update prev_visible_ids for the next frame
            print("updating previous visible IDs...")
            self.prev_visible_ids = current_visible_ids

            for obj_id in disappeared:
                print(f"Object ID {obj_id} disappeared, saving best frame...")
                self._save_best_and_cleanup(obj_id)

        return False  # Do not forward frame downstream

    def _save_best_and_cleanup(self, object_id):
        print(f"Saving best frame for object ID {object_id}...")
        if object_id not in self.best_frames:
            self._clear_object_state(object_id)
            return

        data = self.best_frames[object_id]
        meta = data["metadata"]
        img = data["image"]
        fmt = meta.get("img_format", "BGRx")

        # label = meta.get("label", "")


        filename = f"frame_{self.frame_id}.jpg"

        # Inject mapping field into metadata so that JSON can map to JPG frame
        meta["frame_index"] = self.frame_id
        meta["frame_filename"] = filename

        self.save_image(img, filename, meta)
        self.save_metadata(meta)
        self.frame_id += 1

    def _clear_object_state(self, object_id):
        self.best_scores.pop(object_id, None)
        self.best_frames.pop(object_id, None)
        self.best_meta.pop(object_id, None)


    def save_best_frames(self):
        output_dir = "/tmp/best_frames"
        os.makedirs(output_dir, exist_ok=True)

        for oid, data in self.best_frames.items():
            meta = data["metadata"]
            img = data["image"]
            fmt = meta.get("img_format", "BGRx")

            # Build filename
            label = meta.get("label", "")
            filename = f"frame_{self.frame_id}.jpg"

            # Inject mapping field into metadata so that JSON can map to JPG frame
            meta["frame_index"] = self.frame_id
            meta["frame_filename"] = filename

            # Save image using your helper
            self.save_image(img, filename, meta)

            # Append metadata to saved_metadata list
            self.saved_metadata.append(meta)

            self.frame_id += 1

        logger.info(f"Saved {len(self.best_frames)} best frames to {output_dir}")
        self.best_frames.clear()

        return True  # Forward event downstream

    def save_image(self, image_data, image_filename, metadata):
        # Ensure output directory exists
        output_dir = "/tmp/best_frames"
        os.makedirs(output_dir, exist_ok=True)

        # Convert BGR/BGRx/BGRA to RGB if needed
        if metadata.get("img_format") in ["BGR", "BGRx", "BGRA"]:
            image_data = image_data[:, :, 2::-1]

        # Create PIL image
        image = Image.fromarray(image_data)

        # Build full path
        full_path = os.path.join(output_dir, image_filename)

        try:
            # Save image as JPEG with quality 85
            image.save(full_path, format="JPEG", quality=85)
            logger.info(f"Image saved successfully at {full_path}")
        except Exception as e:
            logger.error(f"Failed to save image {full_path}: {e}")

    def save_metadata(self, metadata):
        metadata_output_path = "/tmp/frames_metadata.json"

        try:
            # Write to JSON file
            with self._lock:
                with open(metadata_output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metadata))
                    f.write("\n")

                logger.info(f"Metadata saved successfully at {metadata_output_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_gva_metadata(self, messages:list) -> dict:
        """Takes a list of frame meta messages, loads them as a JSON and
        updates the metadata dict with the loaded JSON.
        """

        metadata: dict = {}
        for message in messages:
            message_json = json.loads(message)
            metadata.update(message_json)

        return metadata