from .config import config
from .logger import logger
from .chunking import get_chunker
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


def _is_markdown_file(file_path) -> bool:
    return bool(file_path) and str(file_path).lower().endswith(".md")


def _split_markdown_by_headers(docs):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "header_2"),
        ]
    )

    split_docs = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        base_metadata = dict(doc.metadata or {})
        markdown_sections = header_splitter.split_text(text)

        for section in markdown_sections:
            section_text = (section.page_content or "").strip()
            if not section_text:
                continue

            metadata = dict(base_metadata)
            metadata.update(dict(section.metadata or {}))
            if "header_2" in metadata:
                section_text = f"## {metadata['header_2']}\n\n{section_text}"
            split_docs.append(Document(page_content=section_text, metadata=metadata))

    logger.info("Markdown header split created %d sections.", len(split_docs))
    return split_docs


def _split_text_to_indexed_chunks(docs, chunk_size: int, chunk_overlap: int, chunking_strategy: str = "recursive"):
    splitter = get_chunker(
        strategy=chunking_strategy,
        chunk_size=max(50, int(chunk_size)),
        chunk_overlap=max(0, int(chunk_overlap)),
    )

    chunked_docs = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        base_metadata = dict(doc.metadata or {})
        source = str(base_metadata.get("source", "unknown"))
        chunks = splitter.split_text(text)
        total_chunks = len(chunks)

        for index, chunk_text in enumerate(chunks):
            chunk_text = (chunk_text or "").strip()
            if not chunk_text:
                continue

            metadata = dict(base_metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "chunk_size": int(chunk_size),
                    "chunk_overlap": int(chunk_overlap),
                    "source_doc_id": source,
                    "source_total_chunks": total_chunks,
                }
            )
            chunked_docs.append(Document(page_content=chunk_text, metadata=metadata))

    logger.info(
        "Context-window ingestion created %d indexed chunks.",
        len(chunked_docs),
    )
    return chunked_docs


def _index_presplit_docs(docs, chunk_size: int, chunk_overlap: int):
    indexed_docs = []
    total_chunks = len(docs)

    for index, doc in enumerate(docs):
        text = (doc.page_content or "").strip()
        if not text:
            continue

        metadata = dict(doc.metadata or {})
        source = str(metadata.get("source", "unknown"))
        metadata.update(
            {
                "chunk_index": index,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "source_doc_id": source,
                "source_total_chunks": total_chunks,
            }
        )
        indexed_docs.append(Document(page_content=text, metadata=metadata))

    logger.info(
        "Markdown ingestion created %d aisle-level indexed chunks.",
        len(indexed_docs),
    )
    return indexed_docs


def split_documents_for_ingestion(
    docs,
    chunk_size: int,
    chunk_overlap: int,
    file_path=None,
    embedding_model=None,
    chunking_strategy: str = None,
):
    """
    Split source docs into sequential, indexed chunks for context-window retrieval.

    This follows the notebook-style approach:
    1) Chunk each document with overlap.
    2) Store chunk index metadata.
    3) Enrich retrieval later by fetching neighboring indices.

    Args:
        docs: List of LangChain Document objects to split.
        chunk_size: Target chunk size in characters (or tokens for "token" strategy).
        chunk_overlap: Overlap between consecutive chunks.
        file_path: Source file path (used for logging only).
        embedding_model: Unused; kept for backward-compatible caller signature.
        chunking_strategy: Chunking strategy name. If None, falls back to
            ``config.CHUNKING_STRATEGY``. Supported values: recursive, fixed,
            token, section.
    """

    # Unused in this strategy, kept for backward compatibility with caller signature.
    _ = embedding_model

    strategy = chunking_strategy if chunking_strategy is not None else config.CHUNKING_STRATEGY

    logger.info(
        "Ingestion mode: indexed sequential chunking (strategy=%s, file=%s)",
        strategy,
        file_path,
    )

    indexed_chunk_size = max(50, int(chunk_size))
    indexed_chunk_overlap = max(0, min(int(chunk_overlap), indexed_chunk_size // 2))

    if _is_markdown_file(file_path):
        logger.info("Applying MarkdownHeaderTextSplitter with aisle-level chunking.")
        presplit_docs = _split_markdown_by_headers(docs)
        return _index_presplit_docs(
            presplit_docs,
            chunk_size=indexed_chunk_size,
            chunk_overlap=indexed_chunk_overlap,
        )

    return _split_text_to_indexed_chunks(
        docs,
        chunk_size=indexed_chunk_size,
        chunk_overlap=indexed_chunk_overlap,
        chunking_strategy=strategy,
    )
