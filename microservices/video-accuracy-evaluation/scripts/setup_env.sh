#!/bin/bash
# Default values
MODEL_CACHE_PATH="/home/${USER}/model_cache/sbert"

# Check if MODEL_CACHE_PATH exists
if [ -e "$MODEL_CACHE_PATH" ]; then
    # If it exists, check the owner
    if [ "$(stat -c '%U:%G' "$MODEL_CACHE_PATH")" != "root:root" ]; then
        echo "$MODEL_CACHE_PATH exists in host..."
    else
        # If owned by root:root, delete and recreate it
        echo "$MODEL_CACHE_PATH exists and is owned by root:root. Deleting it and recreate..."
        sudo rm -rf "$MODEL_CACHE_PATH"
        mkdir -p "$MODEL_CACHE_PATH"
    fi
else
    # If it doesn't exist, create it
    echo "$MODEL_CACHE_PATH does not exist. Creating it..."
    mkdir -p "$MODEL_CACHE_PATH"
fi


export USER_GROUP_ID=$(id -g ${USER})
export SBERT_MODEL_ID="all-mpnet-base-v2"
export MODEL_CACHE_PATH="$MODEL_CACHE_PATH"