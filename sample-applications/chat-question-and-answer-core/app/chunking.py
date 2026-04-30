import re
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)


class SectionTextSplitter:
    """
    Splits text on structural section separators commonly found in formatted
    knowledge-base documents: horizontal rules (━━━ / ════ / ───), and Markdown
    headings (##, ###).

    Sections that fit within chunk_size are merged greedily. Sections that
    exceed chunk_size are kept as-is to avoid discarding content.
    A trailing overlap window is prepended to the next chunk when a split
    occurs, preserving cross-section context.
    """

    _SEPARATOR_RE = re.compile(
        r"(?:(?<=\n)|^)(?:[━═─]{3,}|#{1,3}(?=\s))",
        re.MULTILINE,
    )

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = max(50, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)

    def split_text(self, text: str) -> list:
        parts = [p.strip() for p in self._SEPARATOR_RE.split(text) if p.strip()]
        chunks = []
        buffer = ""

        for part in parts:
            candidate = (buffer + "\n\n" + part).strip() if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                    overlap_tail = buffer[-self.chunk_overlap:] if self.chunk_overlap else ""
                    buffer = (overlap_tail + "\n\n" + part).strip() if overlap_tail else part
                else:
                    # Single section larger than chunk_size; keep intact
                    chunks.append(part)
                    buffer = ""

        if buffer:
            chunks.append(buffer)

        return chunks

    def split_documents(self, documents) -> list:
        result = []
        for doc in documents:
            for chunk in self.split_text(doc.page_content or ""):
                result.append(
                    Document(page_content=chunk, metadata=dict(doc.metadata or {}))
                )
        return result


def get_chunker(strategy: str = "recursive", chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Factory that returns a text splitter for the requested chunking strategy.

    Args:
        strategy: Chunking strategy. One of:
            - "recursive" (default): RecursiveCharacterTextSplitter — good for
              general prose; tries multiple separators in order.
            - "fixed": CharacterTextSplitter — splits on newlines at a fixed
              character size; best for uniform dense text or logs.
            - "token": TokenTextSplitter — splits by LLM token count; useful
              when staying within a model's context window is critical.
            - "section": SectionTextSplitter — splits on structural separators
              (━━━ / ════ / ##); best for structured docs like retail
              knowledge bases with explicit section headers.
        chunk_size: Target chunk size in characters (or tokens for "token"
            strategy). Minimum 50.
        chunk_overlap: Number of overlapping characters (or tokens) between
            consecutive chunks. Minimum 0.

    Returns:
        A splitter instance with at minimum a ``split_text(text) -> list[str]``
        method, compatible with LangChain's splitter interface.

    Raises:
        ValueError: If an unsupported strategy name is provided.
    """
    strategy = (strategy or "recursive").lower().strip()
    chunk_size = max(50, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))

    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "fixed":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
        )
    elif strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "section":
        return SectionTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            "Supported strategies: recursive, fixed, token, section."
        )
