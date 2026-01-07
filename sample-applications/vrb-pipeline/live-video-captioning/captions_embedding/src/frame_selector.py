# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from io import BytesIO
from math import floor
from threading import Lock
from typing import Set, Tuple, Optional
from PIL import Image
import time
import datetime
import json
import logging
import os
import numpy as np


logger = logging.getLogger('FRAME_SELECTOR')
logger.setLevel(logging.DEBUG)


@dataclass
class ObjState:
    best_score: float = -1.0
    best_image: Optional[np.ndarray] = None        # cropped or full
    best_meta: Optional[dict] = None
    last_seen_frame_idx: int = -1
    miss_count: int = 0
    saved: bool = False
    last_bbox: Tuple[int,int,int,int] = (0,0,0,0)
    label: str = ""


class FrameSelector:
    def __init__(self, *args, **kwargs):
        try:
            logger.info("Initializing FrameSelector via gvapython extension...")

            self.get_env_variables()

            # self.interested: list = kwargs.get("interested")
            # self.interested: list = args[1] if len(args) > 1 else []
            self.interested: list = ["person"]

            self.output_dir = args[0] if len(args) > 0 else "/tmp/default_run/"
            os.makedirs(self.output_dir, exist_ok=True)

            # --- Tunables ---
            self.forward_only_on_save = False        # set False to always forward frames
            self.miss_tolerance = 5                  # consecutive misses before saving
            self.save_full_frame = False             # set True to save full frame
            self.crop_margin = 0.05                  # extra margin around ROI (5%)
            self.jpeg_quality = 85

            # --- Runtime ---
            self.frame_idx = 0
            self.frame_id_counter = 1                # for filenames
            self.obj_states: Dict[int, ObjState] = {}
            self.prev_visible_ids = set()
            self._lock = Lock()

            logger.info(f"FrameSelector initialized. output_dir={self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to initialize FrameSelector: {str(e)}")
            raise


    def get_env_variables(self):
        try:
            print("Getting environment variables for FrameSelector...")

        except ValueError:
            logger.error("Port value should be an integer.")
            raise Exception("Port value should be an integer.")


    def _iou(self, a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x1 = max(ax, bx); y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
        iw = max(0, x2 - x1); ih = max(0, y2 - y1)
        inter = iw * ih
        union = aw*ah + bw*bh - inter
        return (inter/union) if union > 0 else 0.0


    def _find_recent_match(self, label, bbox, window=3, iou_thr=0.5):
        """Find a recently disappeared object (same label) whose bbox overlaps the new one."""
        best_id = None; best_iou = 0.0
        # candidates are tracked objects that are currently not visible (miss_count >= 1)
        for obj_id, st in self.obj_states.items():
            if st.label != label or st.miss_count == 0:
                continue
            if self.frame_idx - st.last_seen_frame_idx > window:
                continue
            i = self._iou(st.last_bbox, bbox)
            if i >= iou_thr and i > best_iou:
                best_iou = i; best_id = obj_id
        return best_id


    def process(self, frame):
        self.frame_idx += 1
        selected_this_frame = False

        # Get np_frame + video info
        with frame.data() as np_frame:
            video_info = frame.video_info()
            fmt = video_info.to_caps().get_structure(0).get_value('format')
            width, height = video_info.width, video_info.height

            # Parse metadata safely
            metadata = self._get_gva_metadata(frame.messages())
            objects = metadata.get("objects", [])
            current_visible_ids = set()

            if not objects:
                # no object detectedm skip processing and drop frame
                # print("no objects detected, dropping frame")
                return False

            # Update state for visible objects
            for obj in objects:
                obj_id = obj.get("id")
                if obj_id is None:
                    continue
                #current_visible_ids.add(obj_id)

                # Extract bbox
                bx, by, bw, bh , is_normalized = self._get_bbox(obj, width, height)
                confidence = float(obj.get("detection", {}).get("confidence", 0))
                label = obj.get("detection", {}).get("label", "")

                if self.interested and label not in self.interested:
                    print(f"Skipping object ID {obj_id} as label {label} is not in interested list.")
                    continue

                current_visible_ids.add(obj_id)

                # Score: area * confidence * sharpness factor (simple Laplacian)
                area = max (bw * bh, 1.0)
                roi = self._crop(np_frame, bx, by, bw, bh, width, height, margin=self.crop_margin)
                sharp = self._sharpness(roi)
                score = confidence * area * (1.0 +0.5 * sharp)  # light bias to sharp frames

                st = self.obj_states.get(obj_id)
                if st is None:
                    # Try to re-associate with a recently missing track of the same label
                    match_id = self._find_recent_match(label, (bx,by,bw,bh), window=3, iou_thr=0.5)
                    if match_id is not None:
                        # Move the previous state to this new tracker id
                        st = self.obj_states.pop(match_id)
                        self.obj_states[obj_id] = st
                    else:
                        st = ObjState()
                        self.obj_states[obj_id] = st

                st.last_seen_frame_idx = self.frame_idx
                st.miss_count = 0   # reset when visible
                st.last_bbox = (bx, by, bw, bh)
                st.label = label

                # update best if improved
                if score > st.best_score:
                    st.best_score = score
                    st.best_image = (np_frame.copy() if self.save_full_frame else roi.copy())
                    st.best_meta = {
                        "object_id": obj_id,
                        "label": label,
                        "confidence": confidence,
                        "bbox": {
                            "x": bx,
                            "y": by,
                            "w": bw,
                            "h": bh,
                            "pixels": True
                        },
                        "img_format": fmt,
                        "frame_index": self.frame_idx
                    }
                    st.saved = False    # new best not saved yet

            # Accumulate miss_count for ALL tracked IDs not visible in this frame
            not_visible_ids = set(self.obj_states.keys()) - current_visible_ids
            for obj_id in list(not_visible_ids):
                st = self.obj_states.get(obj_id)
                if not st:
                    continue
                st.miss_count += 1
                if st.miss_count >= self.miss_tolerance and not st.saved and st.best_image is not None:
                    self._save_best(obj_id, st)
                    st.saved = True
                    selected_this_frame = True
                    del self.obj_states[obj_id]


        # Decide whether to forward the frame downstream
        return (selected_this_frame if self.forward_only_on_save else True)


    def _save_best(self, obj_id: int, st: ObjState):
        """Saves the best image and metadata for the given object state."""
        filename = f"frame_{self.frame_id_counter}.jpg"
        st.best_meta["frame_filename"] = filename

        try:
            self._save_image(st.best_image, filename, st.best_meta)
            self._append_metadata(st.best_meta)
            self.frame_id_counter += 1
            logger.info(f"Saved best frame for object ID {obj_id} as {filename}")
        except Exception as e:
            logger.error(f"Failed to save best frame for object ID {obj_id}: {str(e)}")


    def _save_image(self, image_data: np.ndarray, image_filename: str, metadata: dict):
        """Saves the image data as a JPEG file with given filename and metadata."""
        os.makedirs(self.output_dir, exist_ok=True)
        # Convert BGR/BGRx/BGRA -> RGB
        fmt = metadata.get("img_format", "BGRx")
        if fmt in ["BGR", "BGRx", "BGRA"]:
            image_data = image_data[:, :, 2::-1]  # drop alpha if present
        img = Image.fromarray(image_data)
        img.save(os.path.join(self.output_dir, image_filename), format="JPEG", quality=self.jpeg_quality)


    def _append_metadata(self, meta: dict):
        path = os.path.join(self.output_dir, "frames_metadata.json")
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta))
                f.write("\n")

    def _get_gva_metadata(self, messages:list) -> dict:
        """Takes a list of frame meta messages, loads them as a JSON and
        updates the metadata dict with the loaded JSON.
        """

        metadata: dict = {}
        for message in messages:
            message_json = json.loads(message)
            metadata.update(message_json)

        return metadata

    def _get_bbox(self, obj: dict, width: int, height: int) -> Tuple[int, int, int, int, bool]:
        """Extracts bounding box from object metadata."""
        # Try gvametaconvert style x,y,w,h first
        x = obj.get("x"); y = obj.get("y"); w = obj.get("w"); h = obj.get("h")
        # If absent, try detection.bounding_box
        if None in (x, y, w, h):
            bb = obj.get("detection", {}).get("bounding_box", {})
            x = bb.get("x_min", x); y = bb.get("y_min", y)
            w = bb.get("width", w); h = bb.get("height", h)

        is_normalized = False
        vals = [x, y, w, h]
        if all(v is not None for v in vals):
            # Heuristic: if w,h <= 1.0 assume normalized
            if 0 < float(w) <= 1.0 and 0 < float(h) <= 1.0:
                is_normalized = True
                x = int(floor(float(x) * width))
                y = int(floor(float(y) * height))
                w = int(floor(float(w) * width))
                h = int(floor(float(h) * height))
            else:
                x, y, w, h = int(x), int(y), int(w), int(h)
        else:
            # fall back to full image
            x, y, w, h = 0, 0, width, height
        # clamp
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        return x, y, w, h, is_normalized


    def _crop(self, img: np.ndarray, x: int, y: int, w: int, h: int,
              W: int, H: int, margin: float = 0.0) -> np.ndarray:
        if margin > 0:
            dx = int(w * margin)
            dy = int(h * margin)
            x = max(0, x - dx); y = max(0, y - dy)
            w = min(W - x, w + 2 * dx); h = min(H - y, h + 2 * dy)
        return img[y:y + h, x:x + w]


    def _sharpness(self, roi: np.ndarray) -> float:
        # Fast approximate sharpness: variance of horizontal gradient
        if roi.ndim == 3:
            roi = roi[:, :, 0]  # one channel
        grad = np.abs(np.diff(roi.astype(np.float32), axis=1))
        return float(np.var(grad)) / 255.0