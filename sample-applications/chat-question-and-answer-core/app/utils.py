import os
import re
import shutil
import time
# subprocess is used safely without shell=True and with validated input
import subprocess  # nosec B404
import openvino as ov
import openvino.properties as props
from .logger import logger
from huggingface_hub import login, whoami, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from optimum.intel import (
    OVModelForFeatureExtraction,
    OVModelForSequenceClassification,
    OVModelForCausalLM,
)
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer


def get_available_devices():
    """
    Retrieves a list of available devices from the OpenVINO core.
    Returns:
        list: A list of available device names.
    """

    core = ov.Core()
    device_list = core.available_devices

    return device_list


def get_device_property(device: str = ""):
    """
    Retrieves the properties of a specified device.
    Args:
        device (str): The name of the device to query. Defaults to an empty string.
    Returns:
        dict: A dictionary containing the properties of the device. The keys are property names,
            and the values are the corresponding property values. Non-serializable types are
            converted to strings. If a property value cannot be retrieved due to a TypeError,
            it is set to "UNSUPPORTED TYPE".
    """

    properties_dict = {}
    core = ov.Core()
    supported_properties = core.get_property(device, "SUPPORTED_PROPERTIES")

    for property_key in supported_properties:
        if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
            try:
                property_val = core.get_property(device, property_key)

                # Convert non-serializable types to strings
                if not isinstance(property_val, (str, int, float, bool, type(None))):
                    property_val = str(property_val)

            except TypeError:
                property_val = "UNSUPPORTED TYPE"

            properties_dict[property_key] = property_val

    return properties_dict


def login_to_huggingface(token: str):
    """
    Logs in to Hugging Face using the provided token and checks the authenticated user.

    Args:
        token (str): The authentication token for Hugging Face.

    Returns:
        None

    """

    try:
        logger.info("Logging in to Hugging Face...")

        login(token=token)

        # Check the authenticated user
        user_info = whoami()

        if user_info:
            logger.info(f"Logged in successfully as {user_info['name']}")
        else:
            logger.error("Login failed.")
            raise RuntimeError("Login to Hugging Face failed. Please check your token.")

    except HfHubHTTPError as e:
        logger.error(f"Login failed due to Hugging Face Hub error: {e}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error during Hugging Face login: {e}")
        raise e



def download_huggingface_model(model_id: str, cache_dir: str):
    """
    Downloads a model from the Hugging Face Hub and caches it locally.

    Args:
        model_id (str): The identifier of the model repository on Hugging Face Hub.
        cache_dir (str): The directory path where the model should be cached.

    Raises:
        RepositoryNotFoundError: If the specified model repository does not exist.
        HfHubHTTPError: If an HTTP error occurs while accessing the Hugging Face Hub.
        Exception: For any other unexpected errors during the download process.

    Logs:
        Information about the download process, including start, completion, and any errors encountered.
    """

    try:
        logger.info(f"Starting {model_id} model download...")

        # 'main' is the only available revision and already handled in snapshot_download module
        model_path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)  # nosec B615

        logger.info(f"Repository downloaded to: {model_path}")

    except RepositoryNotFoundError as e:
        logger.error(f"Model repository not found: {model_id}. Please check the model ID.")
        raise e

    except HfHubHTTPError as e:
        logger.error(f"Hugging Face Hub HTTP error occurred: {e}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error while downloading model '{model_id}': {e}")
        raise e


def convert_model(model_id: str, cache_dir: str, model_type: str):
    """
    Converts a specified model to OpenVINO format and saves it to the cache directory.

    Args:
        model_id (str): The identifier of the model to be converted.
        cache_dir (str): The directory where the converted model will be saved.
        model_type (str): The type of the model. It can be "embedding", "reranker", or "llm".

    Returns:
        None

    Raises:
        ValueError: If the model_type is not one of "embedding", "reranker", or "llm".

    Notes:
        - If the model has already been converted and exists in the cache directory, the conversion process is skipped.
        - The function uses the Hugging Face `AutoTokenizer` to load and save the tokenizer.
        - The function uses OpenVINO's `convert_tokenizer` and `save_model` to convert and save the tokenizer.
        - Depending on the model_type, the function uses different OpenVINO model classes to convert and save the model.
    """

    if os.path.isdir(cache_dir + "/" + model_id):
        logger.info(f"Optimized {model_id} exist in {cache_dir}. Skip process...")
    else:
        logger.info(f"Converting {model_id} model to OpenVINO format...")
        # 'main' is the only available revision and already handled in huggingfacehub module
        hf_tokenizer = AutoTokenizer.from_pretrained(model_id)  # nosec B615
        hf_tokenizer.save_pretrained(f"{cache_dir}/{model_id}")
        ov_tokenizer = convert_tokenizer(hf_tokenizer, add_special_tokens=False)
        ov.save_model(ov_tokenizer, f"{cache_dir}/{model_id}/openvino_tokenizer.xml")

        if model_type == "embedding":
            embedding_model = OVModelForFeatureExtraction.from_pretrained(
                model_id, export=True
            )
            embedding_model.save_pretrained(f"{cache_dir}/{model_id}")
        elif model_type == "reranker":
            reranker_model = OVModelForSequenceClassification.from_pretrained(
                model_id, export=True
            )
            reranker_model.save_pretrained(f"{cache_dir}/{model_id}")
        elif model_type == "llm":
            llm_model = OVModelForCausalLM.from_pretrained(
                model_id, export=True, weight_format="int8"
            )
            llm_model.save_pretrained(f"{cache_dir}/{model_id}")


def start_ollama_server(cache_dir: str):
    """
    Starts the Ollama server process with the specified cache directory for model storage.
    This function unsets proxy-related environment variables to avoid issues with Ollama,
    sets the `OLLAMA_MODELS` environment variable to the provided cache directory, and
    launches the Ollama server using the resolved executable path.
    Args:
        cache_dir (str): The directory path where Ollama models will be stored.
    Raises:
        FileNotFoundError: If the Ollama executable is not found in the system PATH.
        Exception: If any other unexpected error occurs during server startup.
    """

    # Known limitation: Unset proxy environment variables to avoid issues with Ollama
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    # Set the `OLLAMA_MODELS` to store the Ollama models
    os.environ['OLLAMA_MODELS'] = cache_dir

    try:
        # Resolved full path to avoid partial path execution
        ollama_path = shutil.which("ollama")
        if ollama_path is None:
            raise FileNotFoundError("ollama executable not found in PATH")

        # full path used, not relying on PATH
        # ollama_path is resolved via shutil.which() and not user-controlled
        serve_process = subprocess.Popen([ollama_path, "serve"])  # nosec B603 B607

        # Optional: wait a few seconds to ensure the server starts
        time.sleep(5)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e


def download_ollama_model(model_id: str, model_type: str):
    """
    Downloads and optionally runs an Ollama model using the Ollama CLI.
    This function pulls the specified model from Ollama using the CLI. If the model type is "llm",
    it also runs the model after downloading. Logs progress and errors throughout the process.

    Args:
        model_id (str): The identifier of the Ollama model to download.
        model_type (str): The type of the model (e.g., "llm" for large language models).

    Raises:
        RuntimeError: If the Ollama CLI fails to download or run the model.
        Exception: For any other unexpected errors.
    """

    try:
        logger.info(f"Starting {model_id} model download...")

        # Resolved full path to avoid partial path execution
        ollama_path = shutil.which("ollama")
        if ollama_path is None:
            raise FileNotFoundError("ollama executable not found in PATH")

        # Download the model using Ollama CLI
        # full path used, not relying on PATH
        # ollama_path is resolved via shutil.which() and not user-controlled
        pull_process = subprocess.run(
            [ollama_path, "pull", model_id],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )  # nosec B603 B607

        logger.info(f"Ollama model {model_id} downloaded successfully.")

        # Run and load the model
        # Only aplicable for LLM models
        # Download the model using Ollama CLI
        # full path used, not relying on PATH
        # ollama_path is resolved via shutil.which() and not user-controlled
        if model_type == "llm":
            run_process = subprocess.run(
                [ollama_path, "run", model_id],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )  # nosec B603 B607

            logger.info(f"Ollama model {model_id} is running successfully.")

    except subprocess.CalledProcessError as e:
        # Clean ANSI escape sequences and extract the last meaningful error line
        raw_error = e.stderr or str(e)
        clean_error = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', raw_error) # Remove ANSI codes
        lines = clean_error.strip().splitlines()
        err_message = next((line for line in reversed(lines) if "Error:" in line), lines[-1] if lines else "Unknown error")
        logger.error(f"Error downloading Ollama model {model_id}: {err_message}")
        raise RuntimeError(f"Ollama failed: {err_message}") from e

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e
