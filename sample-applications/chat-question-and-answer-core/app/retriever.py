from .config import config
from .logger import logger
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from hashlib import sha1
import importlib
import os


def retrieve_docs_with_scoring(retriever, question: str, top_k: int = 5):
    """
    Retrieve docs from base retriever, deduplicate, and trim to top_k.
    """

    retrieved_docs = retriever.invoke(question)

    merged = []
    seen = set()

    for doc in list(retrieved_docs):
        signature = sha1((doc.page_content or "").strip().encode("utf-8")).hexdigest()
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(doc)

    return merged[:top_k]


def _build_contextual_compressor(llm=None, reranker=None):
    """
    Build a contextual compressor for ContextualCompressionRetriever.
    Uses LLMChainExtractor when enabled, otherwise falls back to reranker.
    """

    use_llm_extractor = os.getenv("USE_LLM_CHAIN_EXTRACTOR", "false").lower() == "true"

    if use_llm_extractor:
        module_paths = [
            "langchain.retrievers.document_compressors",
            "langchain_classic.retrievers.document_compressors",
        ]

        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                extractor_cls = getattr(module, "LLMChainExtractor", None)
                if extractor_cls is not None:
                    if llm is None:
                        logger.warning("LLMChainExtractor requested but llm is unavailable; falling back to reranker.")
                        break
                    return extractor_cls.from_llm(llm), "llm_chain_extractor"
            except Exception:
                continue

        logger.warning("USE_LLM_CHAIN_EXTRACTOR is enabled but LLMChainExtractor could not be loaded; falling back to reranker.")

    if reranker is None:
        return None, "none"

    return reranker, "reranker"


def build_retriever(vectorstore, llm=None, reranker=None):
    """
    Create and return the configured retriever, optionally wrapped with contextual compression.
    """

    enable_rerank = config.ENABLE_RERANK
    search_method = str(config.SEARCH_METHOD).lower()
    fetch_k = config.FETCH_K

    logger.info(
        "Creating retriever with search method: %s, fetch_k: %d, rerank enabled: %s",
        search_method,
        fetch_k,
        enable_rerank,
    )

    if vectorstore is None:
        return None

    if search_method == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.5,
            },
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": fetch_k,
            },
        )

    if enable_rerank:
        compressor, compressor_name = _build_contextual_compressor(llm=llm, reranker=reranker)
        if compressor is None:
            logger.warning("Rerank enabled but no compressor is available; proceeding without compression.")
            return retriever

        logger.info("Reranker enabled: %s, compressor: %s", enable_rerank, compressor_name)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )

    return retriever
