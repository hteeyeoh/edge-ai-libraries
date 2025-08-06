from .config import config
from .logger import logger
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import os
import re
import shutil
import time
# subprocess is used safely without shell=True and with validated input
import subprocess  # nosec B404


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

    # Set the `OLLAMA_KEEP_ALIVE` to -1 to keep the model loaded in memory
    os.environ['OLLAMA_KEEP_ALIVE'] = "-1"

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


def init_models():
    # Start Ollama server
    start_ollama_server(config._OLLAMA_CACHE_DIR)

    # Download Ollama models
    download_ollama_model(config.EMBEDDING_MODEL_ID, "embedding")
    download_ollama_model(config.LLM_MODEL_ID, "llm")

    # Initialize Embedding with Ollama
    embedding = OllamaEmbeddings(model=config.EMBEDDING_MODEL_ID)

    # Initialize LLM with Ollama
    llm = OllamaLLM(model=config.LLM_MODEL_ID)

    # Ollama doesn't support reranker model
    reranker = None

    return embedding, llm, reranker