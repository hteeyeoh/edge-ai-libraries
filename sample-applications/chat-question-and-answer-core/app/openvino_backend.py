from .config import config
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
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_huggingface import HuggingFacePipeline
import os
import openvino as ov


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


def init_models():
    # login huggingface
    login_to_huggingface(config.HF_ACCESS_TOKEN)

    # Download convert the model to openvino optimized
    download_huggingface_model(config.EMBEDDING_MODEL_ID, config._CACHE_DIR)
    download_huggingface_model(config.RERANKER_MODEL_ID, config._CACHE_DIR)
    download_huggingface_model(config.LLM_MODEL_ID, config._CACHE_DIR)

    # Convert to openvino IR
    convert_model(config.EMBEDDING_MODEL_ID, config._CACHE_DIR, "embedding")
    convert_model(config.RERANKER_MODEL_ID, config._CACHE_DIR, "reranker")
    convert_model(config.LLM_MODEL_ID, config._CACHE_DIR, "llm")

    # Initialize Embedding Model
    embedding = OpenVINOBgeEmbeddings(
        model_name_or_path=f"{config._CACHE_DIR}/{config.EMBEDDING_MODEL_ID}",
        model_kwargs={"device": config.EMBEDDING_DEVICE, "compile": False},
    )
    embedding.ov_model.compile()

    # Initialize Reranker Model
    reranker = OpenVINOReranker(
        model_name_or_path=f"{config._CACHE_DIR}/{config.RERANKER_MODEL_ID}",
        model_kwargs={"device": config.RERANKER_DEVICE},
        top_n=2,
    )

    # Initialize LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id=f"{config._CACHE_DIR}/{config.LLM_MODEL_ID}",
        task="text-generation",
        backend="openvino",
        model_kwargs={
            "device": config.LLM_DEVICE,
            "ov_config": {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "CACHE_DIR": f"{config._CACHE_DIR}/{config.LLM_MODEL_ID}/model_cache",
            },
            "trust_remote_code": True,
        },
        pipeline_kwargs={"max_new_tokens": config.MAX_TOKENS},
    )
    if llm.pipeline.tokenizer.eos_token_id:
        llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id

    return embedding, llm, reranker