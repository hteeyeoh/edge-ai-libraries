# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import datetime
import json
import logging
import os
import numpy as np

from minio_client import MinioClient
from rabbitmq_mqtt_client import RabbitMQMQTTClient
from PIL import Image


logger = logging.getLogger('PUBLISHER')
logger.setLevel(logging.DEBUG)

class Publisher:
    def __init__(self, *args, **kwargs):
        try:
            logger.info("Initializing Publisher via gvapython extension...")

            self.get_env_variables()
            self.topic: str = kwargs.get("topic")
            self.video_identifier: str = kwargs.get("video_identifier")
            self.bucket_name: str = kwargs.get("minio_bucket")

            # Initialize messages dictionary
            self.messages = {}
            self.frame_id = 0

            if not self.topic or not self.video_identifier or not self.bucket_name:
                logger.error("Missing required arguments: topic, video_identifier or bucket_name")
                raise Exception("Missing required arguments: topic, video_identifier or bucket_name")

            # Initialize connection to RabbitMQ and Minio Clients
            logger.info("Connecting to RabbitMQ MQTT Client...")
            # self.rabbitmq_client = RabbitMQMQTTClient(
            #     self.mqtt_host,
            #     self.mqtt_port,
            #     self.mqtt_username,
            #     self.mqtt_passwd
            # )


            self.rabbitmq_client = RabbitMQMQTTClient(
                broker=self.mqtt_host,
                port=self.mqtt_port,             # 1883 (plaintext) or 8883 (TLS)
                username=self.mqtt_username,
                password=self.mqtt_passwd,
                use_tls=False,                   # set True if using 8883
                # tls_kwargs={...},              # provide CA/cert/key if TLS is enabled
                connect_timeout=10.0,
            )


            if not self.rabbitmq_client.is_connected():
                logger.error(f"Failed to connect to RabbitMQ MQTT Broker - {self.mqtt_host}:{self.mqtt_port}")
            #    return

            logger.info("Connecting to Minio Client...")
            self.minio_client = MinioClient.get_client(
                minio_server=self.minio_server,
                access_key=self.minio_username,
                secret_key=self.minio_passwd
            )

            logger.info("Publisher initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Publisher: {str(e)}")
            raise

    def __del__(self):
        if self.rabbitmq_client and self.rabbitmq_client.is_connected():
            self.rabbitmq_client.stop()
            logger.info("Disconnected from RabbitMQ MQTT Broker.")

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
        """Publish frame and metadata to RabbitMQ MQTT Broker."""
        with frame.data() as image:
            video_info = frame.video_info()
            metadata = self.get_gva_metadata(frame.messages())

            if not metadata:
                # No JSON message => no detections (because add-empty-results=false)
                # Skip publishing but keep the pipeline flowing.
                return True

            metadata["frame_id"] = self.frame_id

            # check
            # print(f"Metadata: {metadata}")

            # Include timestamp into metadata if required
            if os.getenv("ADD_TIMESTAMP_TO_METADATA", "").lower() == "true" and "time" not in metadata:
                metadata["time"] = int(datetime.datetime.now(datetime.timezone.utc).timestamp()*1e9)

            self.frame_id += 1

            # Update metadata with image format
            image_format = video_info.to_caps().get_structure(0).get_value('format')
            metadata["img_format"]=image_format

            # Populate image filename
            image_filename = f"{self.video_identifier}_frame_{metadata['frame_id']}.{MinioClient.file_ext['frame']}"

            # Populate metadata filename
            metadata_filename = f"{self.video_identifier}_frame_{metadata['frame_id']}_metadata.{MinioClient.file_ext['metadata']}"

            # Insert image_url into metadata after saving image to Minio
            image_uri = f"/{self.bucket_name}/{image_filename}"

            # Save image and metadata to Minio
            self.save_image(image, image_filename, metadata)
            self.save_metadata(metadata_filename, metadata, image_uri)

            # Construct the message to be published
            self.messages = {
                "frame_id": self.frame_id,
                "image_uri": image_uri,
                "metadata": metadata
            }

            # Publish the message to RabbitMQ MQTT Broker
            message_payload = json.dumps(self.messages)
            ok = self.rabbitmq_client.publish(self.topic, message_payload, qos=1)
            # logger.info(f"Published frame {metadata['frame_id']} to topic '{self.topic}'.")

            if not ok:
                logger.error("Publish failed or not confirmed")
            else:
                logger.info("Publish confirmed by broker (QoS1)")


        return True

    def get_gva_metadata(self, messages:list) -> dict:
        """Takes a list of frame meta messages, loads them as a JSON and
        updates the metadata dict with the loaded JSON.
        """

        metadata: dict = {}
        for message in messages:
            message_json = json.loads(message)
            metadata.update(message_json)

        return metadata

    def save_image(self, image_data, image_filename, metadata):
        # Invert the BGR color space to RGB
        if metadata.get("img_format") in ["BGR", "BGRx", "BGRA"]:
            image_data = image_data[:, :, 2::-1]

        image = Image.fromarray(image_data)

        logger.info("Saving image")
        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG", quality=85)
        MinioClient.save_object(
            self.minio_client,
            self.bucket_name,
            object_name=image_filename,
            data=image_buffer
        )

    def save_metadata(self, metadata_filename, metadata, image_uri):

        annotated_metadata = {
            "frame_id": self.frame_id,
            "image_uri": image_uri,
            "metadata": metadata
        }

        metadata_dump: str = json.dumps(annotated_metadata, indent=4)
        metadata_dump_bytes = metadata_dump.encode()
        length = len(metadata_dump_bytes)

        logger.info("Saving metadata")
        metadata_buffer = BytesIO(metadata_dump_bytes)

        MinioClient.save_object(
            self.minio_client,
            self.bucket_name,
            object_name=metadata_filename,
            data=metadata_buffer,
            length=length
        )