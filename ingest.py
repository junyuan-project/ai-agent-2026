"""Build the RAG vector store from files in documents/."""

from dotenv import load_dotenv

from rag import DOCUMENTS_DIR, VECTOR_STORE_PATH, ingest_documents

load_dotenv()


def main() -> None:
    print(f"Reading documents from: {DOCUMENTS_DIR}")
    store = ingest_documents()
    count = len(store.store)
    print(f"Indexed {count} chunks -> {VECTOR_STORE_PATH}")
    print("Done. Run: py agent.py")


if __name__ == "__main__":
    main()
