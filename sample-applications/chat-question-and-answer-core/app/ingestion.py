from .config import config
from .logger import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
import numpy as np


SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 &/()\-]+$")
SUBSECTION_HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 &/()\-'.+,#]+:\s*$")
SEPARATOR_RE = re.compile(r"^(?:={10,}|-{10,})$")
LIST_ITEM_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
LABELED_MAJOR_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 &/()\-]{1,80}:\s*.+$")

KEY_VALUE_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 /()&'.,\-]{1,80}:\s+.+$")
MD_HEADER_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
MD_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)", re.MULTILINE)
MD_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
MD_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
MD_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
MD_HRULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)


def _looks_structured_document(text: str) -> bool:
    """
    Generic structure detector for mixed customer documents.

    Uses a weighted score from common formatting patterns instead of
    domain-specific keywords.

    Env var:
    - INGESTION_STRUCTURED_SCORE_THRESHOLD (default: 6)
    """

    if not text:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    top_text = "\n".join(lines[:250])

    section_headers = sum(1 for line in lines[:200] if SECTION_HEADER_RE.match(line))
    major_headers = sum(1 for line in lines[:200] if _is_major_section_header(line))
    separator_lines = len(SEPARATOR_RE.findall(top_text))
    list_items = len(LIST_ITEM_RE.findall(top_text))
    key_value_lines = sum(1 for line in lines[:250] if KEY_VALUE_LINE_RE.match(line))
    # Markdown structure signals
    md_headers = len(MD_HEADER_RE.findall(top_text))
    md_lists = len(MD_LIST_RE.findall(top_text))
    md_table_rows = len(MD_TABLE_ROW_RE.findall(top_text))
    md_table_separators = len(MD_TABLE_SEPARATOR_RE.findall(top_text))
    md_fences = len(MD_FENCE_RE.findall(top_text))
    md_hrules = len(MD_HRULE_RE.findall(top_text))

    # Weighted score favors common structured patterns across many doc types.
    score = 0
    if section_headers >= 3:
        score += 2
    if major_headers >= 2:
        score += 2
    if separator_lines >= 3:
        score += 2
    if list_items >= 8:
        score += 2
    if key_value_lines >= 6:
        score += 2
    if md_headers >= 2:
        score += 2
    if md_lists >= 6:
        score += 1
    if md_table_rows >= 2 and md_table_separators >= 1:
        score += 2
    if md_fences >= 2:
        score += 1
    if md_hrules >= 2:
        score += 1

    threshold_raw = os.getenv("INGESTION_STRUCTURED_SCORE_THRESHOLD", "6")
    try:
        threshold = max(1, int(threshold_raw))
    except ValueError:
        threshold = 6

    is_structured = score >= threshold
    logger.info(
        "Structured detection: score=%d threshold=%d structured=%s (sections=%d major=%d separators=%d lists=%d keyvals=%d md_headers=%d md_lists=%d md_table_rows=%d md_table_sep=%d md_fences=%d md_hrules=%d)",
        score,
        threshold,
        is_structured,
        section_headers,
        major_headers,
        separator_lines,
        list_items,
        key_value_lines,
        md_headers,
        md_lists,
        md_table_rows,
        md_table_separators,
        md_fences,
        md_hrules,
    )
    return is_structured


def _extract_labeled_major_header(line: str):
    if not LABELED_MAJOR_HEADER_RE.match(line):
        return None, None

    header, detail = line.split(":", 1)
    header = header.strip()
    detail = detail.strip()
    return header, detail or None


def _is_major_section_header(line: str) -> bool:
    if not SECTION_HEADER_RE.match(line):
        return False

    # Generic major headers are typically uppercase and at least two words.
    return len(line.split()) >= 2


def _build_structured_chunks(doc: Document, chunk_size: int, chunk_overlap: int):
    text = doc.page_content or ""
    lines = text.splitlines()
    base_metadata = dict(doc.metadata or {})

    blocks = []
    current_major = "GENERAL"
    current_major_detail = None
    current_subsection = None
    current_lines = []

    def flush_block():
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append(
                {
                    "major": current_major,
                    "major_detail": current_major_detail,
                    "subsection": current_subsection,
                    "content": content,
                }
            )
        current_lines = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            current_lines.append(raw_line)
            continue

        if SEPARATOR_RE.match(line):
            flush_block()
            continue

        labeled_major, labeled_detail = _extract_labeled_major_header(line)
        if labeled_major:
            flush_block()
            current_major = labeled_major
            current_major_detail = labeled_detail
            current_subsection = None
            current_lines.append(raw_line)
            continue

        if _is_major_section_header(line):
            flush_block()
            current_major = line
            current_major_detail = None
            current_subsection = None
            current_lines.append(raw_line)
            continue

        if SUBSECTION_HEADER_RE.match(line):
            flush_block()
            current_subsection = line[:-1]
            current_lines.append(raw_line)
            continue

        current_lines.append(raw_line)

    flush_block()

    chunked_docs = []
    for block in blocks:
        block_text = block["content"]
        if not block_text:
            continue

        list_heavy = len(LIST_ITEM_RE.findall(block_text)) >= 3
        long_form = len(block_text) >= 5000

        structured_base_size = max(chunk_size, 3600)
        structured_base_overlap = max(chunk_overlap, 600)

        if list_heavy:
            local_size, local_overlap = structured_base_size, structured_base_overlap
            chunk_kind = "list_heavy"
        elif long_form:
            local_size, local_overlap = max(structured_base_size, 4000), max(
                structured_base_overlap, 700
            )
            chunk_kind = "long_form"
        else:
            local_size, local_overlap = structured_base_size, structured_base_overlap
            chunk_kind = "general"

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n  - ", "\n- ", "\n", ". ", " ", ""],
            chunk_size=local_size,
            chunk_overlap=min(local_overlap, max(0, local_size // 4)),
        )

        section_parts = [f"Section: {block['major']}"]
        if block["major_detail"]:
            section_parts.append(f"Section Detail: {block['major_detail']}")
        if block["subsection"]:
            section_parts.append(f"Subsection: {block['subsection']}")
        section_header = " | ".join(section_parts)

        for idx, chunk in enumerate(splitter.split_text(block_text)):
            if not chunk.strip():
                continue

            metadata = dict(base_metadata)
            metadata.update(
                {
                    "section": block["major"],
                    "section_detail": block["major_detail"],
                    "subsection": block["subsection"],
                    "chunk_kind": chunk_kind,
                    "chunk_seq": idx,
                }
            )
            chunked_docs.append(
                Document(page_content=f"{section_header}\n{chunk}", metadata=metadata)
            )

    return chunked_docs


def _recursive_split(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def _split_sentences(text: str):
    if not text:
        return []

    # Lightweight sentence segmentation without external dependencies.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _semantic_split(
    docs,
    embedding_model,
    similarity_threshold,
    max_tokens,
):
    """
    Chunk documents by semantic continuity between adjacent sentence embeddings.
    Falls back to recursive split when embedding_model is unavailable.
    """

    if embedding_model is None:
        logger.warning("Semantic chunking requested but embedding model is unavailable; falling back to recursive split.")
        return []

    final_chunks = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            final_chunks.append(Document(page_content=text, metadata=dict(doc.metadata or {})))
            continue

        try:
            embeddings = embedding_model.embed_documents(sentences)
        except Exception as err:
            logger.warning("Semantic embedding failed for document; falling back to recursive split for this document: %s", err)
            final_chunks.extend(
                _recursive_split([doc], chunk_size=max_tokens * 4, chunk_overlap=max(50, max_tokens // 10))
            )
            continue

        current_chunk = [sentences[0]]
        current_chunk_tokens = len(sentences[0].split())

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = len(sentence.split())
            similarity = _cosine_similarity(embeddings[i - 1], embeddings[i])

            should_split = (
                similarity < similarity_threshold
                or current_chunk_tokens + sentence_tokens > max_tokens
            )

            if should_split:
                final_chunks.append(
                    Document(
                        page_content=" ".join(current_chunk),
                        metadata=dict(doc.metadata or {}),
                    )
                )
                current_chunk = [sentence]
                current_chunk_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_chunk_tokens += sentence_tokens

        if current_chunk:
            final_chunks.append(
                Document(
                    page_content=" ".join(current_chunk),
                    metadata=dict(doc.metadata or {}),
                )
            )

    return final_chunks


def split_documents_for_ingestion(
    docs,
    chunk_size: int,
    chunk_overlap: int,
    file_path=None,
    embedding_model=None,
):
    """
    Split docs using ingestion strategy from override or INGESTION_STRATEGY.

    Accepted strategies:
    - auto: structured detection + recursive fallback
    - recursive: force RecursiveCharacterTextSplitter for all docs
    - semantic: embedding-based sentence chunking
    """

    strategy = config.INGESTION_STRATEGY.strip().lower()

    if strategy not in {"auto", "recursive", "semantic"}:
        logger.warning("Unknown INGESTION_STRATEGY '%s'. Allowed values: auto, recursive, semantic. Falling back to 'auto'.", strategy)
        strategy = "auto"

    logger.info("Ingestion strategy: %s (file=%s)", strategy, file_path)

    if strategy == "recursive":
        return _recursive_split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if strategy == "semantic":
        semantic_threshold = config.SEMANTIC_SIMILARITY_THRESHOLD
        semantic_max_tokens = config.SEMANTIC_MAX_TOKENS

        semantic_splits = _semantic_split(
            docs,
            embedding_model=embedding_model,
            similarity_threshold=semantic_threshold,
            max_tokens=semantic_max_tokens,
        )
        if semantic_splits:
            for c in semantic_splits:
                print("HT: NEW SECTION")
                print(c.page_content)

            return semantic_splits

        return _recursive_split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # auto strategy
    all_splits = []
    recursive_split_docs = []
    for doc in docs:
        if _looks_structured_document(doc.page_content):
            chunks = _build_structured_chunks(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if chunks:
                all_splits.extend(chunks)
            else:
                recursive_split_docs.append(doc)
        else:
            recursive_split_docs.append(doc)

    if recursive_split_docs:
        all_splits.extend(_recursive_split(recursive_split_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    return all_splits
