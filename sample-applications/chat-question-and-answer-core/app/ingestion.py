from .config import config
from .logger import logger
from langchain_core.documents import Document
<<<<<<< HEAD
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


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
=======
from langchain_text_splitters import RecursiveCharacterTextSplitter
>>>>>>> 555cf7a1 (chatqna-core: Implement neighbors retriever)


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


<<<<<<< HEAD
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


=======
>>>>>>> 555cf7a1 (chatqna-core: Implement neighbors retriever)
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

<<<<<<< HEAD
    docs_to_chunk = docs
    if _is_markdown_file(file_path):
        logger.info("Applying MarkdownHeaderTextSplitter with aisle-level chunking.")
        docs_to_chunk = _split_markdown_by_headers(docs)
        return _index_presplit_docs(
            docs_to_chunk,
            chunk_size=indexed_chunk_size,
            chunk_overlap=indexed_chunk_overlap,
        )

    return _split_text_to_indexed_chunks(
        docs_to_chunk,
=======
    return _split_text_to_indexed_chunks(
        docs,
>>>>>>> 555cf7a1 (chatqna-core: Implement neighbors retriever)
        chunk_size=indexed_chunk_size,
        chunk_overlap=indexed_chunk_overlap,
    )
