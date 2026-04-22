from .config import config
from .logger import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _split_text_to_indexed_chunks(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
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


def split_documents_for_ingestion(
    docs,
    chunk_size: int,
    chunk_overlap: int,
    file_path=None,
    embedding_model=None,
):
    """
    Split source docs into sequential, indexed chunks for context-window retrieval.

    This follows the notebook-style approach:
    1) Chunk each document with overlap.
    2) Store chunk index metadata.
    3) Enrich retrieval later by fetching neighboring indices.
    """

    # Unused in this strategy, kept for backward compatibility with caller signature.
    _ = embedding_model

    logger.info("Ingestion mode: indexed sequential chunking (file=%s)", file_path)

    indexed_chunk_size = max(50, int(chunk_size))
    indexed_chunk_overlap = max(0, min(int(chunk_overlap), indexed_chunk_size // 2))

    return _split_text_to_indexed_chunks(
        docs,
        chunk_size=indexed_chunk_size,
        chunk_overlap=indexed_chunk_overlap,
    )
