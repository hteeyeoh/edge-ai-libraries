from .config import config
from .logger import logger
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from hashlib import sha1


class _BM25RetrieverWithVectorstore:
    """
    Adapter that preserves BM25 retrieval behavior while exposing vectorstore
    for downstream context-window enrichment.
    """

    def __init__(self, bm25_retriever, vectorstore):
        self.bm25_retriever = bm25_retriever
        self.vectorstore = vectorstore

    def invoke(self, query):
        return self.bm25_retriever.invoke(query)

    def __getattr__(self, item):
        return getattr(self.bm25_retriever, item)


def retrieve_enriched_context_docs(retriever, question: str, top_k: int = 5):
    """
    Retrieve query-relevant chunks, expand each with neighbor windows, and
    return the final context documents after optional rerank and deduplication.
    """

    seed_retriever = _extract_seed_retriever(retriever)
    retrieved_docs = list(seed_retriever.invoke(question))
    num_neighbors = max(0, int(config.CONTEXT_WINDOW_NEIGHBORS))
    chunk_overlap = max(0, int(config.CHUNK_OVERLAP))

    logger.info(
        "Context window retrieval: seed_docs=%d top_k=%d neighbors=%d",
        len(retrieved_docs),
        top_k,
        num_neighbors,
    )

    if not retrieved_docs:
        return []

    vectorstore = _extract_vectorstore(retriever)
    if vectorstore is None:
        logger.warning("Context window enrichment skipped: vectorstore not found on retriever. Falling back to deduplicated retrieval.")
        deduped = _deduplicate_docs(retrieved_docs, top_k=None)
        reranked = _apply_post_window_rerank(retriever, deduped, question, top_k)
        return reranked[:top_k]

    if num_neighbors == 0:
        deduped = _deduplicate_docs(retrieved_docs, top_k=None)
        reranked = _apply_post_window_rerank(retriever, deduped, question, top_k)
        return reranked[:top_k]

    all_docs = _get_all_vectorstore_docs(vectorstore)
    if not all_docs:
        deduped = _deduplicate_docs(retrieved_docs, top_k=None)
        reranked = _apply_post_window_rerank(retriever, deduped, question, top_k)
        return reranked[:top_k]

    index_map = _build_neighbor_index_map(all_docs)
    source_bounds = _build_source_chunk_bounds(index_map)
    if not index_map:
        logger.warning("Context window enrichment skipped: required metadata (source/chunk_index) not found. Falling back to deduplicated retrieval.")
        deduped = _deduplicate_docs(retrieved_docs, top_k=None)
        reranked = _apply_post_window_rerank(retriever, deduped, question, top_k)
        return reranked[:top_k]

    merged = []
    seen_signatures = set()
    skipped_no_index = 0

    for doc in list(retrieved_docs):
        metadata = dict(doc.metadata or {})
        source = str(metadata.get("source", metadata.get("source_doc_id", "")))
        chunk_index = metadata.get("chunk_index")

        if not source or chunk_index is None:
            skipped_no_index += 1
            signature = sha1((doc.page_content or "").strip().encode("utf-8")).hexdigest()
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            merged.append(doc)
            continue

        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            skipped_no_index += 1
            signature = sha1((doc.page_content or "").strip().encode("utf-8")).hexdigest()
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            merged.append(doc)
            continue

        enriched_doc = _build_context_window_doc(
            index_map=index_map,
            source=source,
            center_index=chunk_index,
            num_neighbors=num_neighbors,
            source_bounds=source_bounds,
            chunk_overlap=chunk_overlap,
            fallback_doc=doc,
        )

        signature = sha1((enriched_doc.page_content or "").strip().encode("utf-8")).hexdigest()
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        merged.append(enriched_doc)

    if skipped_no_index:
        logger.warning(
            "Context window fallback used for %d seed docs without chunk_index/source metadata. Re-ingest documents with latest ingestion format.",
            skipped_no_index,
        )

    logger.info("Context window retrieval: enriched_docs=%d", len(merged))
    reranked = _apply_post_window_rerank(retriever, merged, question, top_k)
    deduped = _drop_near_duplicate_docs(reranked)
    return deduped[:top_k]


def _extract_seed_retriever(retriever):
    # For context-window logic we need original chunk metadata. If retriever is wrapped
    # by contextual compression, seed on its base retriever to preserve metadata.
    base_retriever = getattr(retriever, "base_retriever", None)
    if base_retriever is not None:
        return base_retriever

    return retriever


def _extract_base_compressor(retriever):
    if retriever is None:
        return None

    return getattr(retriever, "base_compressor", None)


def _extract_vectorstore(retriever):
    if retriever is None:
        return None

    if hasattr(retriever, "vectorstore"):
        return retriever.vectorstore

    base_retriever = getattr(retriever, "base_retriever", None)
    if base_retriever is not None:
        return _extract_vectorstore(base_retriever)

    return None


def _get_all_vectorstore_docs(vectorstore):
    try:
        # Prefer raw docstore iteration so we don't depend on similarity_search ordering
        # or embedding behavior for an empty query.
        docstore = getattr(vectorstore, "docstore", None)
        backing_dict = getattr(docstore, "_dict", None)
        if isinstance(backing_dict, dict) and backing_dict:
            return list(backing_dict.values())

        total = getattr(getattr(vectorstore, "index", None), "ntotal", 0)
        if not total:
            return []
        return vectorstore.similarity_search("", k=total)
    except Exception as err:
        logger.warning("Failed to fetch full vectorstore docs for context enrichment: %s", err)
        return []


def _build_neighbor_index_map(all_docs):
    index_map = {}

    for doc in all_docs:
        metadata = dict(doc.metadata or {})
        source = str(metadata.get("source", metadata.get("source_doc_id", "")))
        chunk_index = metadata.get("chunk_index")
        if not source or chunk_index is None:
            continue

        try:
            normalized_chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            continue

        index_map[(source, normalized_chunk_index)] = doc

    return index_map


def _build_source_chunk_bounds(index_map):
    bounds = {}

    for source, chunk_index in index_map.keys():
        current = bounds.get(source)
        if current is None:
            bounds[source] = [chunk_index, chunk_index]
            continue

        if chunk_index < current[0]:
            current[0] = chunk_index
        if chunk_index > current[1]:
            current[1] = chunk_index

    return bounds


def _build_context_window_doc(
    index_map,
    source,
    center_index,
    num_neighbors,
    source_bounds,
    chunk_overlap,
    fallback_doc,
):
    effective_neighbors = max(0, int(num_neighbors))

    source_min, source_max = source_bounds.get(source, (0, center_index))

    start_index = max(source_min, center_index - effective_neighbors)
    end_index = min(source_max, center_index + effective_neighbors)
    selected_indices = [
        idx
        for idx in range(start_index, end_index + 1)
        if (source, idx) in index_map
    ]
    neighbors = [index_map[(source, i)] for i in selected_indices if (source, i) in index_map]

    if not neighbors:
        return fallback_doc

    neighbors.sort(key=lambda d: int(d.metadata.get("chunk_index", 0)))

    concatenated = neighbors[0].page_content or ""
    for i in range(1, len(neighbors)):
        next_chunk = neighbors[i].page_content or ""
        concatenated = _merge_neighbor_chunks(concatenated, next_chunk, chunk_overlap)

    selected_start = int(selected_indices[0])
    selected_end = int(selected_indices[-1])

    metadata = dict(fallback_doc.metadata or {})
    metadata.update(
        {
            "window_source": source,
            "window_center_index": center_index,
            "window_start_index": selected_start,
            "window_end_index": selected_end,
            "window_num_neighbors": effective_neighbors,
        }
    )
    return Document(page_content=concatenated, metadata=metadata)


def _merge_neighbor_chunks(current_text: str, next_text: str, max_expected_overlap: int) -> str:
    current_text = current_text or ""
    next_text = next_text or ""

    if not current_text:
        return next_text
    if not next_text:
        return current_text

    max_overlap = min(len(current_text), len(next_text), max(0, int(max_expected_overlap)) * 2)
    matched_overlap = 0

    for overlap in range(max_overlap, 0, -1):
        if current_text.endswith(next_text[:overlap]):
            matched_overlap = overlap
            break

    if matched_overlap > 0:
        return current_text + next_text[matched_overlap:]

    # No reliable overlap match found; preserve readability instead of slicing blindly.
    if current_text.endswith("\n") or next_text.startswith("\n"):
        return current_text + next_text

    return current_text + "\n" + next_text


def _deduplicate_docs(retrieved_docs, top_k):
    merged = []
    seen_signatures = set()

    for doc in list(retrieved_docs):
        signature = sha1((doc.page_content or "").strip().encode("utf-8")).hexdigest()
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        merged.append(doc)

    if top_k is None:
        return merged

    return merged[:top_k]


def _apply_post_window_rerank(retriever, docs, question: str, top_k: int):
    compressor = _extract_base_compressor(retriever)
    if compressor is None:
        return docs

    try:
        reranked_docs = compressor.compress_documents(docs, question)
        reranked_list = list(reranked_docs)
        rerank_top_n = max(1, int(getattr(config, "RERANK_TOP_N", top_k)))
        min_target = min(max(1, int(top_k)), rerank_top_n, len(docs))

        # Keep rerank ordering, then backfill with remaining expanded windows when
        # compressor unexpectedly returns too few docs.
        if len(reranked_list) < min_target:
            seen_signatures = {
                sha1((doc.page_content or "").strip().encode("utf-8")).hexdigest()
                for doc in reranked_list
            }
            for candidate in docs:
                signature = sha1((candidate.page_content or "").strip().encode("utf-8")).hexdigest()
                if signature in seen_signatures:
                    continue
                reranked_list.append(candidate)
                seen_signatures.add(signature)
                if len(reranked_list) >= min_target:
                    break

        logger.info(
            "Post-window rerank/compression applied: input_docs=%d output_docs=%d target_min=%d",
            len(docs),
            len(reranked_list),
            min_target,
        )
        return reranked_list
    except Exception as err:
        logger.warning("Post-window rerank/compression failed; returning uncompressed docs: %s", err)
        return docs


def _drop_near_duplicate_docs(docs):
    if not docs:
        return docs

    deduped = []
    seen_signatures = set()

    for doc in docs:
        text = " ".join((doc.page_content or "").split())
        if not text:
            continue

        signature = _sampled_text_signature(text)
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        deduped.append(doc)

    return deduped


def _sampled_text_signature(text: str) -> str:
    if not text:
        return ""

    text_len = len(text)
    head = text[:500]
    mid_start = max(0, (text_len // 2) - 150)
    mid = text[mid_start: mid_start + 300]
    tail = text[-500:] if text_len > 500 else text

    # Sample across the document to suppress repeated boilerplate blocks while
    # preserving windows whose content differs in the middle section.
    sampled = "|".join((head, mid, tail))
    return sha1(sampled.encode("utf-8")).hexdigest()


def _build_contextual_compressor(llm=None, reranker=None):
    """
    Build a contextual compressor for ContextualCompressionRetriever.
    Uses LLMChainExtractor when enabled, otherwise falls back to reranker.
    """

    if reranker is None:
        return None, "none"

    if config.ENABLE_LLM_CHAIN_EXTRACTOR:
        compressor = LLMChainExtractor.from_llm(llm)
        logger.info("Using LLMChainExtractor as contextual compressor.")
        return compressor, "llm_chain_extractor"

    logger.info("Using reranker as contextual compressor.")
    return reranker, "reranker"



def build_retriever(vectorstore, llm=None, reranker=None):
    """
    Create and return the configured retriever, optionally wrapped with contextual compression.
    """

    enable_rerank = config.ENABLE_RERANK
    search_method = str(config.SEARCH_METHOD).lower()
    fetch_k = config.FETCH_K
    score_threshold = config.SIMILARITY_SCORE_THRESHOLD

    logger.info(
        "Creating retriever with search method: %s, fetch_k: %d, score_threshold: %.3f, rerank enabled: %s",
        search_method,
        fetch_k,
        score_threshold,
        enable_rerank,
    )

    if vectorstore is None:
        return None

    if search_method == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": fetch_k,
                "fetch_k": 20,
                "lambda_mult": 0.5,
            },
        )
    elif search_method == "bm25":
        all_docs = _get_all_vectorstore_docs(vectorstore)
        if not all_docs:
            logger.warning("BM25 retriever requested but vectorstore docs are unavailable; falling back to similarity retriever.")
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": fetch_k,
                },
            )
        else:
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = fetch_k
            # Keep vectorstore accessible for context-window neighbor enrichment.
            retriever = _BM25RetrieverWithVectorstore(bm25_retriever, vectorstore)
    elif search_method == "similarity_score_threshold":
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": fetch_k,
                "score_threshold": score_threshold,
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
