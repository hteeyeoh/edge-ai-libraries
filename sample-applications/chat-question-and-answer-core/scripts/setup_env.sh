#!/bin/bash

# This script sets up the environment for the Chat Question and Answer Core application.
# It accepts two optional parameters:
# -p or --path: Specify the model cache path (default is /tmp/model_cache/)
# Usage:
# ./setup_env.sh
# OR
# ./setup_env.sh -p /path/to/model_cache

# GPUs currently tested:
# - Arc A770 dGPU
# - Battlemage G21 dGPU
# - Arrowlake iGPU

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_CACHE_PATH="/home/${USER}/model_cache/"
PROFILES="OPENVINO"
BACKEND="OPENVINO"
BACKEND_HOST="chatqna-core-openvino"
DEFAULT_OPENVINO_MODEL_CONFIG_PATH="${PROJECT_ROOT}/model_config/sample/openvino_template.yaml"
DEFAULT_OLLAMA_MODEL_CONFIG_PATH="${PROJECT_ROOT}/model_config/sample/ollama_template.yaml"

print_help() {
    echo "Usage: source setup.sh [options]"
    echo "Options:"
    echo "  -p, --path <path>       Specify the model cache path"
    echo "  -b, --backend <backend> Specify the backend (openvino or ollama)"
    echo "  -h, --help              Display this help message"
}

# Parse named arguments using getopts
# -p for MODEL_CACHE_PATH
# -b for BACKEND
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--path) MODEL_CACHE_PATH="$2"; shift ;;
        -b|--backend) BACKEND="$2"; shift ;;
        -h|--help)
            print_help
            return 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            print_help
            return 1
            ;;
    esac
    shift
done


# Convert BACKEND to uppercase to handle both uppercase and lowercase inputs
BACKEND=$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')
# Check if BACKEND value is valid
if [[ "$BACKEND" != "OPENVINO" && "$BACKEND" != "OLLAMA" ]]; then
    echo "Error: Unsupported backend: '$BACKEND'. Supported backends are 'openvino' or 'ollama'."
    return 1
fi

# Resolve model configuration path. Priority:
# 1) Existing MODEL_CONFIG_PATH from environment
# 2) Backend-specific default template path
if [[ -z "${MODEL_CONFIG_PATH}" ]]; then
    if [[ "$BACKEND" == "OPENVINO" ]]; then
        MODEL_CONFIG_PATH="$DEFAULT_OPENVINO_MODEL_CONFIG_PATH"
    else
        MODEL_CONFIG_PATH="$DEFAULT_OLLAMA_MODEL_CONFIG_PATH"
    fi
fi

# Normalize MODEL_CONFIG_PATH to absolute path when a relative path is provided
if [[ "$MODEL_CONFIG_PATH" != /* ]]; then
    MODEL_CONFIG_PATH="$PWD/$MODEL_CONFIG_PATH"
    echo "Relative MODEL_CONFIG_PATH provided. Using absolute path: $MODEL_CONFIG_PATH"
fi

if [[ ! -f "$MODEL_CONFIG_PATH" ]]; then
    echo "Error: MODEL_CONFIG_PATH file not found: $MODEL_CONFIG_PATH"
    return 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required to parse MODEL_CONFIG_PATH: $MODEL_CONFIG_PATH"
    return 1
fi

read_model_config_value() {
    local field_name="$1"
    python3 - "$MODEL_CONFIG_PATH" "$field_name" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
field_name = sys.argv[2]

with open(config_path, "r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}

value = (data.get("model_settings") or {}).get(field_name, "")
if value is None:
    value = ""

print(str(value))
PY
}

read_device_config_value() {
    local field_name="$1"
    python3 - "$MODEL_CONFIG_PATH" "$field_name" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
field_name = sys.argv[2]

with open(config_path, "r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}

value = (data.get("device_settings") or {}).get(field_name, "")
if value is None:
    value = ""

print(str(value))
PY
}

MODEL_RUNTIME_FROM_CONFIG="$(read_model_config_value "MODEL_RUNTIME")"
if [[ -z "$MODEL_RUNTIME_FROM_CONFIG" ]]; then
    echo "Error: MODEL_RUNTIME is missing in model_settings of $MODEL_CONFIG_PATH"
    return 1
fi

MODEL_RUNTIME_FROM_CONFIG_UPPER="$(echo "$MODEL_RUNTIME_FROM_CONFIG" | tr '[:lower:]' '[:upper:]')"
if [[ "$MODEL_RUNTIME_FROM_CONFIG_UPPER" != "$BACKEND" ]]; then
    echo "Error: MODEL_RUNTIME in $MODEL_CONFIG_PATH is '$MODEL_RUNTIME_FROM_CONFIG' but BACKEND is '$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')'."
    echo "Please make them match (openvino/ollama) and run setup again."
    return 1
fi

if [[ "$BACKEND" == "OPENVINO" ]]; then
    EMBEDDING_MODEL_ID="$(read_model_config_value "EMBEDDING_MODEL_ID")"
    LLM_MODEL_ID="$(read_model_config_value "LLM_MODEL_ID")"
    RERANKER_MODEL_ID="$(read_model_config_value "RERANKER_MODEL_ID")"
    if [[ -z "$EMBEDDING_MODEL_ID" || -z "$LLM_MODEL_ID" || -z "$RERANKER_MODEL_ID" ]]; then
        echo "Error: EMBEDDING_MODEL_ID, LLM_MODEL_ID and RERANKER_MODEL_ID must be set in model_settings of $MODEL_CONFIG_PATH for OPENVINO backend"
        return 1
    fi

    EMBEDDING_DEVICE="$(read_device_config_value "EMBEDDING_DEVICE")"
    RERANKER_DEVICE="$(read_device_config_value "RERANKER_DEVICE")"
    LLM_DEVICE="$(read_device_config_value "LLM_DEVICE")"
    EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-CPU}"
    RERANKER_DEVICE="${RERANKER_DEVICE:-CPU}"
    LLM_DEVICE="${LLM_DEVICE:-CPU}"
fi

start_model_download_service() {
    local md_container_name="$1"
    local md_image="$2"
    local md_port="$3"
    local md_model_root="$4"
    local md_ovms_tag="${OVMS_TAG:-${MODEL_DOWNLOAD_OVMS_TAG:-v2026.4.1}}"
    local md_hf_token="${HUGGINGFACE_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"

    if ! command -v docker >/dev/null 2>&1; then
        echo "Error: docker is required to start model-download service."
        return 1
    fi

    mkdir -p "$md_model_root"

    docker rm -f "$md_container_name" >/dev/null 2>&1

    local docker_env_args=(
        -e "MODEL_PATH=/opt/models"
        -e "HF_HUB_ENABLE_HF_TRANSFER=1"
        -e "OVMS_RELEASE_TAG=${md_ovms_tag}"
        -e "UV_CACHE_DIR=/tmp/model-download-cache/uv"
        -e "no_proxy=${no_proxy:-}"
        -e "http_proxy=${http_proxy:-}"
        -e "https_proxy=${https_proxy:-}"
    )

    if [[ -n "$md_hf_token" ]]; then
        docker_env_args+=( -e "HF_TOKEN=${md_hf_token}" )
    fi

    if ! docker run -d \
        --name "$md_container_name" \
        -p "127.0.0.1:${md_port}:8000" \
        "${docker_env_args[@]}" \
        -v "${md_model_root}:/opt/models" \
        "$md_image" \
        --plugins openvino >/dev/null; then
        echo "Error: Failed to start model-download service container '$md_container_name'."
        return 1
    fi

    local elapsed=0
    local health_url="http://127.0.0.1:${md_port}/health"
    echo "INFO - model-download OVMS tag: ${md_ovms_tag}"
    echo "INFO - Waiting for model-download service health check at ${health_url}"
    while true; do
        if curl -s -f --max-time 5 "$health_url" >/dev/null 2>&1; then
            echo "INFO - model-download service is healthy."
            return 0
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "$md_container_name" 2>/dev/null)" != "true" ]]; then
            echo "Error: model-download service container exited before becoming healthy."
            docker rm -f "$md_container_name" >/dev/null 2>&1 || true
            return 1
        fi
        if (( elapsed >= 120 )); then
            echo "Error: model-download service did not become healthy within 120 seconds."
            docker rm -f "$md_container_name" >/dev/null 2>&1 || true
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
}

stop_model_download_service() {
    local md_container_name="$1"
    docker rm -f "$md_container_name" >/dev/null 2>&1 || true
}

submit_model_download_job() {
    local md_api_url="$1"
    local model_name="$2"
    local model_type="$3"
    local precision="$4"
    local device="$5"
    local config_payload payload
    local npu_llm_extra_quantization_params="${NPU_LLM_EXTRA_QUANTIZATION_PARAMS:---group-size 128  --ratio 0.8}"

    if [[ "${device^^}" == "NPU" && ( "$model_type" == "embeddings" || "$model_type" == "rerank" ) ]]; then
        config_payload="{\"precision\":\"${precision}\"}"
    elif [[ "${device^^}" == "NPU" && "$model_type" == "llm" ]]; then
        config_payload="{\"precision\":\"${precision}\",\"device\":\"${device}\",\"extra_quantization_params\":\"${npu_llm_extra_quantization_params}\"}"
    else
        config_payload="{\"precision\":\"${precision}\",\"device\":\"${device}\"}"
    fi

    payload="{\"models\":[{\"name\":\"${model_name}\",\"hub\":\"openvino\",\"type\":\"${model_type}\",\"is_ovms\":true,\"config\":${config_payload}}],\"parallel_downloads\":false}"

    local response_file
    response_file="$(mktemp)"
    local http_code
    http_code="$(curl -s -o "$response_file" -w '%{http_code}' \
        -X POST "${md_api_url}/models/download?download_path=ovms_model" \
        -H "Content-Type: application/json" \
        -d "$payload")"

    if [[ "$http_code" != "200" ]]; then
        echo "Error: model-download API failed for ${model_type} model '${model_name}' (HTTP ${http_code})."
        echo "API response: $(cat "$response_file")"
        rm -f "$response_file"
        return 1
    fi

    local job_id
    job_id="$(python3 - "$response_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

job_ids = payload.get("job_ids") or []
print(job_ids[0] if job_ids else "")
PY
)"

    rm -f "$response_file"

    if [[ -z "$job_id" ]]; then
        echo "Error: model-download API did not return a job id for ${model_type} model '${model_name}'."
        return 1
    fi

    echo "$job_id"
}

wait_model_download_job() {
    local md_api_url="$1"
    local job_id="$2"
    local model_label="$3"

    local elapsed=0
    while true; do
        local status
        status="$(curl -s --max-time 10 "${md_api_url}/jobs/${job_id}" | python3 -c 'import json,sys
try:
    print((json.load(sys.stdin) or {}).get("status", ""))
except Exception:
    print("")')"

        if [[ "$status" == "completed" ]]; then
            echo "INFO - model-download completed for ${model_label} (job: ${job_id})."
            return 0
        fi
        if [[ "$status" == "failed" ]]; then
            echo "Error: model-download failed for ${model_label} (job: ${job_id})."
            return 1
        fi
        if (( elapsed >= 3600 )); then
            echo "Error: model-download timed out for ${model_label} after 3600 seconds (job: ${job_id})."
            return 1
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done
}

get_model_weight_format() {
    local target_device="$1"
    case "${target_device^^}" in
        *NPU*|*GPU*)
            echo "int4"
            ;;
        *)
            echo "int8"
            ;;
    esac
}

get_preconverted_precision_from_model_name() {
    local model_id="$1"
    local short_name
    short_name="${model_id##*/}"
    short_name="${short_name,,}"

    # For pre-converted OpenVINO models, infer precision from the repo name.
    if [[ "$model_id" == OpenVINO/* ]] && [[ "$short_name" =~ -(int4|int8|fp16|fp32)(-|$) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi

    return 1
}

has_openvino_export_files() {
    local model_dir="$1"
    [[ -f "${model_dir}/openvino_model.xml" && -f "${model_dir}/openvino_model.bin" ]]
}

get_ovms_model_path() {
    local model_id="$1"
    local target_device="$2"
    local precision="$3"

    printf '%s/ovms_model/openvino_models/%s/%s/%s' \
        "$MODEL_CACHE_PATH" \
        "$(printf '%s' "$target_device" | sed -E 's/[^A-Za-z0-9._-]+/_/g' | tr '[:upper:]' '[:lower:]')" \
        "$(printf '%s' "$precision" | tr '[:upper:]' '[:lower:]')" \
        "$model_id"
}

sync_cpu_to_npu_if_needed() {
    local model_id="$1"
    local model_type="$2"
    local precision="$3"
    local target_device="$4"

    # model-download service does not support device=NPU for embeddings/rerank.
    # Keep payload without device and copy CPU artifacts to NPU layout.
    if [[ "${target_device^^}" != "NPU" || ( "$model_type" != "embeddings" && "$model_type" != "rerank" ) ]]; then
        return 0
    fi

    local cpu_path npu_path
    cpu_path="$(get_ovms_model_path "$model_id" "CPU" "$precision")"
    npu_path="$(get_ovms_model_path "$model_id" "NPU" "$precision")"

    if has_openvino_export_files "$npu_path"; then
        return 0
    fi

    if has_openvino_export_files "$cpu_path"; then
        mkdir -p "$npu_path"
        cp -a "$cpu_path/." "$npu_path/"
        if has_openvino_export_files "$npu_path"; then
            echo "INFO - Synced ${model_type} model artifacts from CPU path to NPU path for ${model_id}."
            return 0
        fi
    fi

    return 1
}

is_model_already_downloaded() {
    local model_id="$1"
    local target_device="$2"
    local precision="$3"

    # model-download OVMS layout
    local ovms_path
    ovms_path="$(get_ovms_model_path "$model_id" "$target_device" "$precision")"
    if has_openvino_export_files "$ovms_path"; then
        return 0
    fi

    return 1
}

trigger_openvino_model_downloads() {
    local md_container_name="${MODEL_DOWNLOAD_CTR_NAME:-chatqna-model-download}"
    local md_image="${MODEL_DOWNLOAD_IMAGE:-intel/model-download:${MODEL_DOWNLOAD_TAG:-latest}}"
    local md_host_port="${MODEL_DOWNLOAD_HOST_PORT:-8200}"
    local md_api_url="http://127.0.0.1:${md_host_port}/api/v1"

    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required to trigger model-download API."
        return 1
    fi

    local embedding_job_id reranker_job_id llm_job_id
    local embedding_precision reranker_precision llm_precision
    local need_embedding_download=true need_reranker_download=true need_llm_download=true

    embedding_precision="$(get_preconverted_precision_from_model_name "$EMBEDDING_MODEL_ID" || get_model_weight_format "$EMBEDDING_DEVICE")"
    reranker_precision="$(get_preconverted_precision_from_model_name "$RERANKER_MODEL_ID" || get_model_weight_format "$RERANKER_DEVICE")"
    llm_precision="$(get_preconverted_precision_from_model_name "$LLM_MODEL_ID" || get_model_weight_format "$LLM_DEVICE")"

    # For NPU embeddings/rerank, try reusing CPU artifacts by syncing CPU->NPU first.
    sync_cpu_to_npu_if_needed "$EMBEDDING_MODEL_ID" "embeddings" "$embedding_precision" "$EMBEDDING_DEVICE" || true
    sync_cpu_to_npu_if_needed "$RERANKER_MODEL_ID" "rerank" "$reranker_precision" "$RERANKER_DEVICE" || true

    if is_model_already_downloaded "$EMBEDDING_MODEL_ID" "$EMBEDDING_DEVICE" "$embedding_precision"; then
        need_embedding_download=false
        echo "INFO - Found existing OpenVINO export for EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}. Skipping download."
    fi
    if is_model_already_downloaded "$RERANKER_MODEL_ID" "$RERANKER_DEVICE" "$reranker_precision"; then
        need_reranker_download=false
        echo "INFO - Found existing OpenVINO export for RERANKER_MODEL_ID=${RERANKER_MODEL_ID}. Skipping download."
    fi
    if is_model_already_downloaded "$LLM_MODEL_ID" "$LLM_DEVICE" "$llm_precision"; then
        need_llm_download=false
        echo "INFO - Found existing OpenVINO export for LLM_MODEL_ID=${LLM_MODEL_ID}. Skipping download."
    fi

    if [[ "$need_embedding_download" == false && "$need_reranker_download" == false && "$need_llm_download" == false ]]; then
        echo "INFO - All OpenVINO model artifacts already exist. No download needed."
        return 0
    fi

    start_model_download_service "$md_container_name" "$md_image" "$md_host_port" "$MODEL_CACHE_PATH" || return 1

    if [[ "$need_embedding_download" == true ]]; then
        embedding_job_id="$(submit_model_download_job "$md_api_url" "$EMBEDDING_MODEL_ID" "embeddings" "$embedding_precision" "$EMBEDDING_DEVICE")" || {
            stop_model_download_service "$md_container_name"
            return 1
        }
        wait_model_download_job "$md_api_url" "$embedding_job_id" "EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}" || {
            stop_model_download_service "$md_container_name"
            return 1
        }
        sync_cpu_to_npu_if_needed "$EMBEDDING_MODEL_ID" "embeddings" "$embedding_precision" "$EMBEDDING_DEVICE" || {
            echo "Error: Failed to prepare NPU path for EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID} from CPU artifacts."
            stop_model_download_service "$md_container_name"
            return 1
        }
    fi

    if [[ "$need_reranker_download" == true ]]; then
        reranker_job_id="$(submit_model_download_job "$md_api_url" "$RERANKER_MODEL_ID" "rerank" "$reranker_precision" "$RERANKER_DEVICE")" || {
            stop_model_download_service "$md_container_name"
            return 1
        }
        wait_model_download_job "$md_api_url" "$reranker_job_id" "RERANKER_MODEL_ID=${RERANKER_MODEL_ID}" || {
            stop_model_download_service "$md_container_name"
            return 1
        }
        sync_cpu_to_npu_if_needed "$RERANKER_MODEL_ID" "rerank" "$reranker_precision" "$RERANKER_DEVICE" || {
            echo "Error: Failed to prepare NPU path for RERANKER_MODEL_ID=${RERANKER_MODEL_ID} from CPU artifacts."
            stop_model_download_service "$md_container_name"
            return 1
        }
    fi

    if [[ "$need_llm_download" == true ]]; then
        llm_job_id="$(submit_model_download_job "$md_api_url" "$LLM_MODEL_ID" "llm" "$llm_precision" "$LLM_DEVICE")" || {
            stop_model_download_service "$md_container_name"
            return 1
        }
        wait_model_download_job "$md_api_url" "$llm_job_id" "LLM_MODEL_ID=${LLM_MODEL_ID}" || {
            stop_model_download_service "$md_container_name"
            return 1
        }
    fi

    stop_model_download_service "$md_container_name"
}

# Check if MODEL_CACHE_PATH is an absolute path
if [[ "$MODEL_CACHE_PATH" != /* ]]; then
    MODEL_CACHE_PATH="$PWD/$MODEL_CACHE_PATH"
    echo "Relative path provided. Using absolute path: $MODEL_CACHE_PATH"
fi

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

if [[ "$BACKEND" == "OPENVINO" ]]; then
    echo "INFO - Triggering OpenVINO model download for:"
    echo "          EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}"
    echo "          RERANKER_MODEL_ID=${RERANKER_MODEL_ID}"
    echo "          LLM_MODEL_ID=${LLM_MODEL_ID}"
    trigger_openvino_model_downloads || return 1
fi


# Set environment variables based on the backend and device
if [ "$BACKEND" == "OPENVINO" ]; then
    # Check if render device exists
    if compgen -G "/dev/dri/render*" > /dev/null; then
        echo "GPU device found."
        if compgen -G "/dev/accel/accel*" > /dev/null; then
            export NPU_DEVICE_PATH="$(compgen -G "/dev/accel/accel*" | head -n 1)"
            echo "NPU device found."
        fi
    fi
elif [ "$BACKEND" == "OLLAMA" ]; then
    PROFILES="OLLAMA"
    BACKEND_HOST="chatqna-core-ollama"
    echo -e "Current ollama backend doesn't support GPU. Only CPU is supported."
    echo -e "Use CPU processing..."
fi

export USER_GROUP_ID=$(id -g ${USER})
if [ -n "${HUGGINGFACEHUB_API_TOKEN}" ]; then
    export HF_ACCESS_TOKEN="${HUGGINGFACEHUB_API_TOKEN}"
else
    echo "INFO - HUGGINGFACEHUB_API_TOKEN is not set. HF_ACCESS_TOKEN will not be exported. Only publicly available models can be downloaded."
fi
export HOST_IP=$(ip route get 1 | awk '{print $7}')
export MODEL_CACHE_PATH="$MODEL_CACHE_PATH"
export MODEL_CONFIG_PATH="$MODEL_CONFIG_PATH"
export APP_BACKEND_URL="/v1/chatqna"
export COMPOSE_PROFILES=$PROFILES
export UI_HOST="chatqna-core-ui"
export BACKEND_HOST=$BACKEND_HOST

# Generate nginx.conf file for docker compose
source ./nginx_config/generate_nginx_conf.sh
