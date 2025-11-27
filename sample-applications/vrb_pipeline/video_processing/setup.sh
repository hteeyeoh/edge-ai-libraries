#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Get Host IP Address
host_ip=$(ip route get 1 | awk '{print $7}')

export MINIO_SERVICE=minio
export RABBITMQ_SERVICE=rabbitmq

export MINIO_SERVER=${MINIO_SERVICE}:9000
export MINIO_HOST_PORT=9000
export MINIO_CONSOLE_HOST_PORT=9001
export AMQP_HOST_PORT=5672
export RABBITMQ_UI_HOST_PORT=15672
export MQTT_HOST_PORT=1883

if ! [[ $no_proxy == *"${MINIO_SERVICE}"* ]]; then
    export no_proxy="$no_proxy,$MINIO_SERVICE,$RABBITMQ_SERVICE,$host_ip"
fi


# MINIO and RABBITMQ credentials
export MINIO_ROOT_USER=intel
export MINIO_ROOT_PASSWORD=intel123
export RABBITMQ_DEFAULT_USER=intel
export RABBITMQ_DEFAULT_PASS=intel123

echo "All required environment variables set successfully. \
Please make sure Minio and RabbitMQ credentials are set on your shell before proceeding."
