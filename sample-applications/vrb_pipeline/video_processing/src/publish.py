# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import datetime
import json
import logging
import os
import numpy as np
from PIL import Image


logger = logging.getLogger('PUBLISHER')
logger.setLevel(logging.DEBUG)

class Publisher:
    def __init__(self, *args, **kwargs):
        try:
            logger.info("Initializing Publisher via gvapython extension...")

            self.get_env_variables()
            self.interested: list = kwargs.get("interested")
            print(self.interested)

            # Initialize messages dictionary
            self.messages = {}
            self.frame_id = 1

            self.best_frames = {}
            self.saved_metadata = []

            logger.info("Publisher initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Publisher: {str(e)}")
            raise

    def __del__(self):
        """Destructor to clean up resources."""
        print("saving best frames and metadata")
        self.save_best_frames()
        self.save_metadata()
        print("Pipeline finished. Destroying resources...")

    def get_env_variables(self):
        try:
            self.mqtt_host: str = os.getenv("RABBITMQ_HOST", "localhost")
            self.mqtt_port: int = int(os.getenv("RABBITMQ_PORT", "1883"))
            self.mqtt_username: str = os.getenv("RABBITMQ_DEFAULT_USER")
            self.mqtt_passwd: str = os.getenv("RABBITMQ_DEFAULT_PASS")
            self.minio_server: str = os.getenv("MINIO_SERVER", "localhost:9000")
            self.minio_username: str = os.getenv("MINIO_ROOT_USER")
            self.minio_passwd: str = os.getenv("MINIO_ROOT_PASSWORD")

        except ValueError:
            logger.error("Port value should be an integer.")
            raise Exception("Port value should be an integer.")


    def process(self, frame):
        with frame.data() as image:
            video_info = frame.video_info()
            metadata = self.get_gva_metadata(frame.messages())

            if not metadata:
                return False  # Drop frames with no detections

            # Add timestamp if enabled
            if os.getenv("ADD_TIMESTAMP_TO_METADATA", "").lower() == "true" and "time" not in metadata:
                metadata["time"] = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)

            objects = metadata.get("objects", [])
            if not objects:
                return False  # Drop frames with no objects

            img_format = video_info.to_caps().get_structure(0).get_value('format')

            for obj in objects:
                obj_id = obj.get("id")
                if obj_id is None:
                    continue

                obj_label = obj.get("detection", {}).get("label", "")

                if self.interested:
                    if obj_label not in self.interested:
                        print(f"Skipping object with label: {obj_label}")
                        continue

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
                            "img_format": img_format
                        },
                        "image": image.copy()  # Store full frame
                    }


            return False  # Drop all frames; we only save at EOS


    def save_best_frames(self):
        print("Saving best frames to disk...")
        output_dir = "/tmp/best_frames"
        os.makedirs(output_dir, exist_ok=True)

        for obj_id, data in self.best_frames.items():
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

    def save_metadata(self):
        metadata_output_path = "/tmp/best_frames_metadata.json"
        print(f"Metadatato be saved: {self.saved_metadata}")
        try:
            # Write to JSON file
            with open(metadata_output_path, "w") as f:
                print("trying to save")
                json.dump(self.saved_metadata, f, indent=4)

                logger.info(f"Metadata saved successfully at {metadata_output_path}")

            self.saved_metadata.clear()

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