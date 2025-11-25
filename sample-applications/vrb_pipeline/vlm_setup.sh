#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

export HOST_IP=$(hostname -I | awk '{print $1}')
export no_proxy_env=${no_proxy}

export VLM_DEVICE=${VLM_DEVICE:-CPU}

export TAG=${TAG:-latest}

# If REGISTRY_URL is set, ensure it ends with a trailing slash
# Using parameter expansion to conditionally append '/' if not already present
[[ -n "$REGISTRY_URL" ]] && REGISTRY_URL="${REGISTRY_URL%/}/"

# If PROJECT_NAME is set, ensure it ends with a trailing slash
[[ -n "$PROJECT_NAME" ]] && PROJECT_NAME="${PROJECT_NAME%/}/"

export REGISTRY="${REGISTRY_URL}${PROJECT_NAME}"

docker volume create ov-models
echo "Created docker volume for the models."

export VLM_COMPRESSION_WEIGHT_FORMAT=${VLM_COMPRESSION_WEIGHT_FORMAT:-int8}
# Number of uvicorn workers
export WORKERS=${WORKERS:-1}

if [[ "$VLM_DEVICE" == "GPU" ]]; then
    export VLM_COMPRESSION_WEIGHT_FORMAT=int4
    export WORKERS=1
fi

# Export current user and group IDs for container user
export USER_ID=$(id -u)
export USER_GROUP_ID=$(id -g)
export VIDEO_GROUP_ID=$(getent group video | awk -F: '{printf "%s\n", $3}')
export RENDER_GROUP_ID=$(getent group render | awk -F: '{printf "%s\n", $3}')

export VLM_SERVICE_PORT=${VLM_SERVICE_PORT:-9764}
export VLM_SEED=${VLM_SEED:-42}
export VLM_LOG_LEVEL=${VLM_LOG_LEVEL:-info}

# VLM_ACCESS_LOG_FILE: Controls where Gunicorn access logs go
# - "-" for stdout (default)
# - "/dev/null" to disable access logs (recommended for production health checks)
# - file path to write access logs to a file

# Set VLM_OPENVINO_LOG_LEVEL based on VLM_LOG_LEVEL
# OpenVINO log levels: 0=NO, 1=ERR, 2=WARNING, 3=INFO, 4=DEBUG, 5=TRACE
case "${VLM_LOG_LEVEL}" in
    "debug")
        export VLM_OPENVINO_LOG_LEVEL=4  # DEBUG
        export VLM_ACCESS_LOG_FILE=${VLM_ACCESS_LOG_FILE:--}
        ;;
    "info")
        export VLM_OPENVINO_LOG_LEVEL=0  # INFO
        export VLM_ACCESS_LOG_FILE=${VLM_ACCESS_LOG_FILE:-/dev/null}
        ;;
    "warning")
        export VLM_OPENVINO_LOG_LEVEL=2  # WARNING
        export VLM_ACCESS_LOG_FILE=${VLM_ACCESS_LOG_FILE:--}
        ;;
    "error")
        export VLM_OPENVINO_LOG_LEVEL=1  # ERR
        export VLM_ACCESS_LOG_FILE=${VLM_ACCESS_LOG_FILE:--}
        ;;
    *)
        export VLM_OPENVINO_LOG_LEVEL=0  # INFO (default)
        export VLM_ACCESS_LOG_FILE=${VLM_ACCESS_LOG_FILE:-/dev/null}
        ;;
esac

# By default, VLM_MAX_COMPLETION_TOKENS is unset (which results in None in Python)
# To set a specific value, either export it before sourcing this script or uncomment the following line:
# export VLM_MAX_COMPLETION_TOKENS=1000
if [ -z "$VLM_MAX_COMPLETION_TOKENS" ]; then
    unset VLM_MAX_COMPLETION_TOKENS
else
    export VLM_MAX_COMPLETION_TOKENS=$VLM_MAX_COMPLETION_TOKENS
    echo -e "\nVLM_MAX_COMPLETION_TOKENS is set: ${VLM_MAX_COMPLETION_TOKENS}"
fi

# OpenVINO Configuration (optional)
# OV_CONFIG allows you to pass OpenVINO configuration parameters as a JSON string
# Common parameters include:
# - PERFORMANCE_HINT: "LATENCY" (default) or "THROUGHPUT"
# - CACHE_DIR: Path to cache directory
# - NUM_STREAMS: Number of inference streams
# - INFERENCE_NUM_THREADS: Number of inference threads
# - And many more OpenVINO configuration options
#
# Examples:
# export OV_CONFIG='{"PERFORMANCE_HINT": "THROUGHPUT"}'
# export OV_CONFIG='{"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 2}'
# export OV_CONFIG='{"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "/tmp/ov_cache"}'
# export OV_CONFIG='{"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_NUM_THREADS": 8, "NUM_STREAMS": 4}'
#
# If not set, the default configuration will be: {"PERFORMANCE_HINT": "LATENCY"}
# To set a specific value, either export it before sourcing this script or uncomment one of the examples above
if [ -z "$OV_CONFIG" ]; then
    unset OV_CONFIG
    echo -e "\nOV_CONFIG is not set. Using default OpenVINO configuration."
    echo "To set custom OpenVINO configuration, export OV_CONFIG before sourcing this script."
    echo "Example: export OV_CONFIG='{\"PERFORMANCE_HINT\": \"THROUGHPUT\"}'"
else
    export OV_CONFIG=$OV_CONFIG
    echo -e "\nOV_CONFIG is set: ${OV_CONFIG}"
fi

# By default, HUGGINGFACE_TOKEN is unset
# To download Gated Models you need to pass your Huggingface Token as described here (https://huggingface.co/docs/hub/models-gated#access-gated-models-as-a-user)
# To set a specific value, uncomment and modify the following line:
# export HUGGINGFACE_TOKEN=<your_huggingface_token_here>
# Only unset HUGGINGFACE_TOKEN if it is not already set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    unset HUGGINGFACE_TOKEN
    echo -e "\nWARNING: HUGGINGFACE_TOKEN is not set."
    echo "Some models on Hugging Face require authentication to download (e.g., gated or private models)."
    echo "If you need to access such models, you must set the HUGGINGFACE_TOKEN environment variable."
    echo "To obtain your token:"
    echo "  1. Visit https://huggingface.co/settings/tokens and log in."
    echo "  2. Create or copy your existing access token."
    echo "  3. Set it in your shell: export HUGGINGFACE_TOKEN=<your_huggingface_token_here>"
    echo "  4. Then re-run this setup script."
else
    echo -e "HUGGINGFACE_TOKEN is set."
fi

# Check if VLM_MODEL_NAME is not defined or empty
if [ -z "$VLM_MODEL_NAME" ]; then
    echo -e "\nERROR: VLM_MODEL_NAME is not set in your shell environment.\n"
    return
else
    export VLM_MODEL_NAME=$VLM_MODEL_NAME
    echo -e "\nVLM_MODEL_NAME: ${VLM_MODEL_NAME}"
fi

# Multimodal embedding serving
export EMBEDDING_USE_OV=false
export EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-CPU}
export OV_PERFORMANCE_MODE=${OV_PERFORMANCE_MODE:-LATENCY}

# If EMBEDDING_DEVICE is GPU, set EMBEDDING_USE_OV to true
if [ "$EMBEDDING_DEVICE" = "GPU" ]; then
    export EMBEDDING_USE_OV=true
fi

export EMBEDDING_SERVER_PORT=9777
export EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}

# Model path configuration
export EMBEDDING_OV_MODELS_DIR=${EMBEDDING_OV_MODELS_DIR:-"/app/ov_models"}

# Check if EMBEDDING_MODEL_NAME is supported
case "$EMBEDDING_MODEL_NAME" in
    "CLIP/clip-vit-b-16"|"CLIP/clip-vit-l-14"|"CLIP/clip-vit-b-32"|"CLIP/clip-vit-h-14")
        echo "Using CLIP model: $EMBEDDING_MODEL_NAME"
        ;;
    "CN-CLIP/cn-clip-vit-b-16"|"CN-CLIP/cn-clip-vit-l-14"|"CN-CLIP/cn-clip-vit-h-14")
        echo "Using CN-CLIP model: $EMBEDDING_MODEL_NAME (Chinese + English support)"
        ;;
    "SigLIP/siglip2-vit-b-16"|"SigLIP/siglip2-vit-l-16"|"SigLIP/siglip2-so400m-patch16-384")
        echo "Using SigLIP model: $EMBEDDING_MODEL_NAME"
        ;;
    "MobileCLIP/mobileclip_s0"|"MobileCLIP/mobileclip_s1"|"MobileCLIP/mobileclip_s2"|"MobileCLIP/mobileclip_b"|"MobileCLIP/mobileclip_blt")
        echo "Using MobileCLIP model: $EMBEDDING_MODEL_NAME"
        ;;
    "Blip2/blip2_transformers")
        echo "Using BLIP2 model: $EMBEDDING_MODEL_NAME"
        ;;
    *)
        echo -e "WARNING: Model '$EMBEDDING_MODEL_NAME' may not be supported."
        echo -e "See docs/user-guide/supported-models.md for the complete list of supported models."
        ;;
esac

# video processing default
export DEFAULT_START_OFFSET_SEC=0
export DEFAULT_CLIP_DURATION=-1  # -1 means take the video till end
export DEFAULT_NUM_FRAMES=64

echo "EMBEDDING_MODEL_NAME set to: ${EMBEDDING_MODEL_NAME}"
echo "EMBEDDING_DEVICE set to: ${EMBEDDING_DEVICE}"
echo "EMBEDDING_USE_OV set to: ${EMBEDDING_USE_OV}"
echo "OV_PERFORMANCE_MODE set to: ${OV_PERFORMANCE_MODE}"


# env for vdms-vector-db
export VDMS_VDB_HOST_PORT=55555
export VDMS_VDB_HOST=vdms-vector-db

# env for rabbitmq
export CONFIG_DIR=${PWD}/config
export RABBITMQ_CONFIG=${CONFIG_DIR}/rmq.conf
export RABBITMQ_AMQP_HOST_PORT=5672
export RABBITMQ_MANAGEMENT_UI_HOST_PORT=15672
export RABBITMQ_MQTT_HOST_PORT=1883
export RABBITMQ_USER=guest
export RABBITMQ_PASSWORD=guest123