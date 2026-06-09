"""RAG: load documents, embed with Gemini, search via vector store."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vectorstore.json"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def _load_text_files(documents_dir: Path) -> list[tuple[str, str]]:
    """Return (source_path, text) for each supported file."""
    if not documents_dir.is_dir():
        return []

    files: list[tuple[str, str]] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            rel = path.relative_to(documents_dir).as_posix()
            files.append((rel, text))
    return files


def ingest_documents(
    documents_dir: Path | None = None,
    *,
    vector_store_path: Path | None = None,
) -> InMemoryVectorStore:
    """Chunk documents, embed, and persist the vector store."""
    documents_dir = documents_dir or DOCUMENTS_DIR
    vector_store_path = vector_store_path or VECTOR_STORE_PATH

    raw_files = _load_text_files(documents_dir)
    if not raw_files:
        raise FileNotFoundError(
            f"No .txt or .md files found in {documents_dir}. "
            "Add documents there, then run: py ingest.py"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    texts: list[str] = []
    metadatas: list[dict] = []
    for source, content in raw_files:
        for chunk in splitter.split_text(content):
            texts.append(chunk)
            metadatas.append({"source": source})

    embeddings = get_embeddings()
    store = InMemoryVectorStore.from_texts(texts, embeddings, metadatas=metadatas)

    vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(vector_store_path))
    return store


def load_vector_store(
    vector_store_path: Path | None = None,
) -> InMemoryVectorStore | None:
    """Load persisted vector store, or None if not built yet."""
    vector_store_path = vector_store_path or VECTOR_STORE_PATH
    if not vector_store_path.is_file():
        return None
    return InMemoryVectorStore.load(str(vector_store_path), get_embeddings())


def format_search_results(docs: list) -> str:
    if not docs:
        return "No relevant passages found in the knowledge base."

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def make_search_tool(vector_store: InMemoryVectorStore):
    """Create an agent tool that searches the knowledge base."""

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search uploaded documents for information relevant to the query.

        Use this when the user asks about topics that may be in their private
        documents, notes, or knowledge base. Pass a focused search query.
        """
        docs = vector_store.similarity_search(query, k=TOP_K)
        return format_search_results(docs)

    return search_knowledge_base
