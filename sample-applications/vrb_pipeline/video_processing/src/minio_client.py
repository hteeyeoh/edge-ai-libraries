# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from minio import Minio
from minio.error import S3Error
import pathlib


class MinioClient:
    """Creates a singleton Minio Client Object and contains helper methods
    to work with Minio Server.
    """

    client = None

    file_ext = {
        "frame": "jpeg",
        "metadata": "json"
    }

    @classmethod
    def get_client(cls, minio_server: str, access_key: str, secret_key: str) -> Minio:
        """Returns an object of Minio Client if none exists already"""
        try:
            if not cls.client:
                cls.client = Minio(
                    minio_server,
                    access_key,
                    secret_key,
                    secure=False,
                )
            return cls.client

        except S3Error as ex:
            raise ex

    @staticmethod
    def ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
        """ Ensure that the given bucket exists on minio server.
        If not, create the bucket.
        """
        try:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)

        except S3Error as err:
            raise Exception(f"Error ocurred during bucket creation: {err}")

    staticmethod
    def save_object(client: Minio, bucket_name: str, object_name: str, data: BytesIO, length: int = 0) -> None:
        """ Save the provided data as a resource on minio at the given bucket name.
        """

        # Ensure the bucket exists
        MinioClient.ensure_bucket_exists(client, bucket_name)

        if not length:
            length = data.tell()
            data.seek(0)

        try:
            client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data,
                length=length,
                content_type="application/octet-stream"
            )

        except S3Error as err:
            raise Exception(f"Error ocurred during saving to bucket: {err}")
