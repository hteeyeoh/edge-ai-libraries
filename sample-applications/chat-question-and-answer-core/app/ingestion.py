from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

from .config import config
from .logger import logger


def _recursive_split(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def _normalize_file_paths(file_path=None, docs=None):
    if file_path is None:
        file_path = []
    elif isinstance(file_path, (str, Path)):
        file_path = [file_path]
    elif isinstance(file_path, Iterable):
        file_path = list(file_path)
    else:
        file_path = [file_path]

    normalized = [str(Path(path)) for path in file_path if path]
    if normalized:
        return normalized

    for doc in docs or []:
        source = (doc.metadata or {}).get("source")
        if source:
            normalized.append(str(Path(source)))

    # Keep order but remove duplicates.
    seen = set()
    deduped = []
    for path in normalized:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def split_documents_for_ingestion(
    docs,
    chunk_size: int,
    chunk_overlap: int,
    file_path=None,
    embedding_model=None,
):
    """
    Split documents by delegating parsing/chunking to langchain-unstructured.
    Falls back to recursive splitting only when unstructured loading fails.
    """

    del embedding_model

    paths = _normalize_file_paths(file_path=file_path, docs=docs)
    if not paths:
        logger.warning(
            "No file path available for UnstructuredLoader. Falling back to recursive split."
        )
        return _recursive_split(docs or [], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    max_characters = int(getattr(config, "UNSTRUCTURED_MAX_CHARACTERS", chunk_size) or chunk_size)
    new_after_n_chars = int(
        getattr(config, "UNSTRUCTURED_NEW_AFTER_N_CHARS", max_characters)
        or max_characters
    )
    combine_under_n_chars = int(
        getattr(config, "UNSTRUCTURED_COMBINE_TEXT_UNDER_N_CHARS", chunk_overlap)
        or chunk_overlap
    )

    chunking_strategy = config.UNSTRUCTURED_CHUNKING_STRATEGY.lower()
    if chunking_strategy == "by_title":
        loader_kwargs = {
            "file_path": paths,
            "chunking_strategy": getattr(config, "UNSTRUCTURED_CHUNKING_STRATEGY", "by_title"),
            "strategy": getattr(config, "UNSTRUCTURED_STRATEGY", "fast"),
            "max_characters": max(1, max_characters),
            "new_after_n_chars": max(1, new_after_n_chars),
            "combine_text_under_n_chars": max(0, combine_under_n_chars),
            "overlap": 300,
            "overlap_all": True
        }
    elif chunking_strategy == "basic":
        loader_kwargs = {
            "file_path": paths,
            "chunking_strategy": getattr(config, "UNSTRUCTURED_CHUNKING_STRATEGY", "by_title"),
            "strategy": getattr(config, "UNSTRUCTURED_STRATEGY", "fast"),
            "max_characters": max(1, max_characters),
            "new_after_n_chars": max(1, new_after_n_chars),
            "combine_text_under_n_chars": max(0, combine_under_n_chars),
            "overlap": 200,
        }


    logger.info(
        "Ingestion via UnstructuredLoader (files=%d, chunking_strategy=%s, strategy=%s)",
        len(paths),
        loader_kwargs["chunking_strategy"],
        loader_kwargs["strategy"],
    )

    try:
        loader = UnstructuredLoader(**loader_kwargs)
        loaded_docs = loader.load()
        if loaded_docs:
            return loaded_docs

        logger.warning(
            "UnstructuredLoader returned no chunks. Falling back to recursive split."
        )
    except Exception as err:
        logger.warning(
            "UnstructuredLoader failed. Falling back to recursive split: %s",
            err,
        )

    return _recursive_split(docs or [], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
