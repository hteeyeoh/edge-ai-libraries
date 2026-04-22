from .config import config
from .document import load_file_document
from .ingestion import split_documents_for_ingestion
from .logger import logger
from .prompt import CONDENSE_QUESTION_TEMPLATE
from .retriever import build_retriever
from .retriever import retrieve_enriched_context_docs
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
import os
import importlib
import re
import pandas as pd

vectorstore = None

# The RUN_TEST flag is used to bypass the model download and conversion steps during pytest unit testing.
# If RUN_TEST is set to "True", the model download and conversion steps are skipped.
# This flag is set in the conftest.py file before running the tests.
if os.getenv("RUN_TEST", "").lower() != "true":
    if config.MODEL_RUNTIME == "openvino":
        runtime_module = importlib.import_module("app.openvino_backend")
        runtime_instance = runtime_module.OpenVINOBackend()

    elif config.MODEL_RUNTIME == "ollama":
        runtime_module = importlib.import_module("app.ollama_backend")
        runtime_instance = runtime_module.OllamaBackend()

    else:
        raise ValueError(f"Unsupported model runtime: {config.MODEL_RUNTIME}")

    embedding, llm, reranker = runtime_instance.init_models()

    template = config.PROMPT_TEMPLATE

    prompt = ChatPromptTemplate.from_template(template)
    condense_question_prompt = ChatPromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    condense_question_chain = condense_question_prompt | llm | StrOutputParser()

else:
    logger.info("Bypassing to mock these functions because RUN_TEST is set to 'True' to run pytest unit test.")


def default_context(_):
    """
    Returns a default context when the retriever is None.

    This function is used to provide a default context in scenarios where
    the retriever is not available or not provided.

    Returns:
        str: An empty string as the default context.
    """

    return ""


def _parse_history_turns(history: str):
    """
    Parse history text into ordered (role, content) turns.
    Supports formats like: "user: ... assistant: ..." across one or multiple lines.
    """

    if not history:
        return []

    pattern = re.compile(r"\b(user|assistant)\s*:\s*", flags=re.IGNORECASE)
    matches = list(pattern.finditer(history))
    if not matches:
        return []

    turns = []
    for idx, match in enumerate(matches):
        role = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(history)
        content = history[start:end].strip()
        if content:
            turns.append((role, content))

    return turns


def _build_condense_question_buffer(history: str):
    """
    Keep only recent turns for follow-up question condensation.
    """
    max_turns = config._CONDENSE_BUFFER_TURNS

    turns = _parse_history_turns(history)
    if not turns:
        return ""

    recent_turns = turns[-max_turns:]
    return "\n".join(f"{role}: {content}" for role, content in recent_turns)


def _rewrite_question_from_history(question: str, history: str):
    """
    Expand shorthand follow-up queries with prior intent from chat history.
    """

    q = (question or "").strip()
    if not q:
        return q

    condensed_buffer = _build_condense_question_buffer(history)

    # Normalize conversational follow-up prefixes.
    q = re.sub(
        r"^(?:how\s+about|what\s+about|and|about)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()
    if not q:
        return question

    # Heuristic for short follow-ups like "how about chia seeds?"
    is_short = len(q.split()) <= 5
    has_where_intent = any(token in q.lower() for token in ["where", "which aisle", "find", "get"])

    if has_where_intent:
        return q

    if is_short and condensed_buffer:
        lower_history = condensed_buffer.lower()
        if any(token in lower_history for token in ["where", "aisle", "find", "get"]):
            return f"Where can I find {q.rstrip('?')}?"

    return q


def _rewrite_question_with_llm(question: str, history: str):
    """
    Rewrite follow-up user query to a standalone question using an LLM.
    Falls back to None on any runtime failure.
    """

    if not config.ENABLE_LLM_CONDENSE_QUESTION:
        return None

    chain = globals().get("condense_question_chain")
    if chain is None:
        return None

    q = (question or "").strip()
    if not q:
        return q

    condensed_buffer = _build_condense_question_buffer(history)
    if not condensed_buffer:
        # Do not run condenser for the first turn or empty history.
        return None

    try:
        rewritten = chain.invoke(
            {
                "history": condensed_buffer,
                "question": q,
            },
            config={"callbacks": []},
        )
    except Exception as err:
        logger.debug("LLM condense rewrite failed: %s", err)
        return None

    rewritten = _sanitize_condensed_question(rewritten, q)
    if not rewritten:
        return None
    return rewritten


def _sanitize_condensed_question(raw_text, original_question: str):
    """
    Normalize condenser output and reject prompt-echo/instructional text.
    """

    text = (raw_text or "").strip().strip('"').strip("'")
    if not text:
        return None

    lower = text.lower()
    if "standalone question:" in lower:
        text = re.split(r"standalone\s+question\s*:\s*", text, flags=re.IGNORECASE)[-1].strip()

    # Keep the last non-empty line; many models echo prompts then append output.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[-1]

    text = re.sub(r"^(human|assistant|ai)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip('"').strip("'")

    bad_markers = [
        "given the recent conversation",
        "latest user question",
        "standalone question",
        "rewrite the latest question",
    ]
    if any(marker in text.lower() for marker in bad_markers):
        return None

    if len(text) < 3 or text.endswith(":"):
        return None

    # If condenser returns unchanged question, treat as a valid no-op.
    return text


def _get_effective_question(question: str, history: str):
    """
    Resolve standalone user question for retrieval and prompt input.
    Prefer LLM condensation, fallback to heuristic rewrite.
    """

    effective_question, _ = _get_effective_question_with_source(question, history)
    return effective_question


def _get_effective_question_with_source(question: str, history: str):
    """
    Resolve standalone user question and include rewrite source for observability.
    """

    llm_rewrite = _rewrite_question_with_llm(question, history)
    if llm_rewrite:
        original = (question or "").strip()
        source = "llm" if llm_rewrite != original else "none"
        return llm_rewrite, source

    heuristic_rewrite = _rewrite_question_from_history(question, history)
    original = (question or "").strip()
    source = "heuristic" if heuristic_rewrite != original else "none"
    return heuristic_rewrite, source


def _prepare_rag_chain_input(chain_input, retriever):
    """
    Prepare context/question/history in one step to avoid duplicate condense calls.
    """

    original_question = chain_input.get("question", "")
    history = chain_input.get("history", "")
    effective_question, rewrite_source = _get_effective_question_with_source(
        original_question, history
    )

    if config.ENABLE_LLM_CONDENSE_QUESTION:
        prompt_question = effective_question or original_question
    else:
        prompt_question = original_question

    docs = retrieve_enriched_context_docs(retriever, effective_question)
    context_text = "\n\n".join(doc.page_content for doc in docs)
    _log_rendered_prompt(context_text, prompt_question, history)

    return {
        "context": context_text,
        "question": prompt_question,
        "history": history,
    }


def _log_rendered_prompt(context_text: str, question: str, history: str):
    """
    Log the final rendered prompt text that is sent to the model.
    """

    try:
        rendered_prompt = prompt.format_prompt(
            context=context_text,
            question=question,
            history=history,
        ).to_string()
    except Exception:
        rendered_prompt = prompt.format(
            context=context_text,
            question=question,
            history=history,
        )

    logger.info("RAG final prompt payload:\n%s", rendered_prompt)


def get_retriever():
    """
    Creates and returns a retriever object with optional reranking capability.

    Returns:
        retriever: A retriever object, optionally wrapped with a contextual compression reranker.

    """

    return build_retriever(vectorstore=vectorstore, llm=llm, reranker=reranker)


def build_chain(retriever=None):
    """
    Builds a Retrieval-Augmented Generation (RAG) chain using the provided retriever.

    Args:
        retriever: A retriever object that fetches relevant documents based on a query.

    Returns:
        A RAG chain that processes the context and question, and generates a response.
    """

    if retriever:
        prepare_input = RunnableLambda(
            lambda chain_input: _prepare_rag_chain_input(chain_input, retriever)
        )

        chain = prepare_input | prompt | llm | StrOutputParser()

    else:
        chain = (
            {
                "context": default_context,
                "question": lambda chain_input: chain_input.get("question", ""),
                "history": lambda chain_input: chain_input.get("history", ""),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    return chain


async def process_query(chain=None, chain_input=None):
    """
    Processes a query using the provided chain and yields the results asynchronously.
    Args:
        chain: An optional chain object that has an `astream` method to process the query.
        chain_input (dict): A dictionary containing the question and optional history.
    Yields:
        str: The processed data chunks in the format "data: {chunk}\n\n".
    """

    if chain_input is None:
        chain_input = {"question": "", "history": ""}

    async for chunk in chain.astream(chain_input):
        yield f"data: {chunk}\n\n"


def create_faiss_vectordb(file_path: str = ""):
    """
    Creates a FAISS vector database from a document file.
    This function loads a document from the specified file path, splits it into chunks,
    creates embeddings for the chunks, and stores them in a FAISS vector database. If a
    global vectorstore already exists, it merges the new embeddings into the existing
    vectorstore.

    Args:
        file_path (str): The path to the document file. Defaults to an empty string.
        chunk_size (int): The size of each chunk in characters. Defaults to 1000.
        chunk_overlap (int): The number of overlapping characters between chunks. Defaults to 200.

    Returns:
        bool: True if the vector database was created successfully.
    """

    global vectorstore
    chunk_size = config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP

    # Load the document from the /tmp path and create embedding
    docs = load_file_document(file_path)
    splits = split_documents_for_ingestion(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        file_path=file_path,
        embedding_model=globals().get("embedding"),
    )

    if not splits:
        logger.error("No text data from the document.")
        return False

    doc_embedding = FAISS.from_documents(documents=splits, embedding=embedding)
    if vectorstore == None:
        vectorstore = doc_embedding
    else:
        vectorstore.merge_from(doc_embedding)

    return True


def get_document_from_vectordb():
    """
    Retrieve document names from the vector database.
    This function accesses the global `vectorstore` object, extracts document
    metadata, and returns a list of document names.

    Returns:
        []: Return empty list if the `vectorstore` is None.
        list: A list of document names extracted from the vector database.
    """

    global vectorstore

    if vectorstore is None:
        return []

    vstore = vectorstore.docstore._dict

    docs = {vstore[key].metadata["source"].split("/")[-1] for key in vstore.keys()}

    return list(docs)


def delete_embedding_from_vectordb(document: str = "", delete_all: bool = False):
    """
    Deletes embeddings from the vector database.

    Args:
        document (str): The name of the document whose embeddings are to be deleted. If empty, no specific document is targeted.
        delete_all (bool): If True, all embeddings in the vector database will be deleted. If False, only embeddings related to the specified document will be deleted.

    Returns:
        bool: True if the deletion process completes successfully.
    """

    global vectorstore

    if vectorstore is None:
        return False

    vstore = vectorstore.docstore._dict
    data_rows = []

    for key in vstore.keys():
        doc_name = vstore[key].metadata["source"].split("/")[-1]
        content = vstore[key].page_content
        data_rows.append(
            {
                "chunk_id": key,
                "document": doc_name,
                "content": content,
            }
        )

    vectordf = pd.DataFrame(data_rows)

    if delete_all:
        # delete all the embeddings in vectorstore
        chunk_list = vectordf["chunk_id"].tolist()
    else:
        # delete the specified document embeddings in vectorstore
        chunk_list = vectordf.loc[vectordf["document"] == document]["chunk_id"].tolist()

    vectorstore.delete(chunk_list)

    return True
