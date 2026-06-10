# RAG Chat Agent — Retrieval-Augmented Chat (RAG) CLI

A production-oriented retrieval-augmented chat assistant that indexes local markdown/text documents, embeds them with Google Gemini embeddings, and provides a durable, multi-thread CLI chat experience. This project demonstrates backend engineering, LLM/tool integration, vector search pipelines, and production-grade persistence—ideal.

---

## Elevator Pitch

Built a robust RAG pipeline: document ingestion, chunking, Gemini embeddings, persisted vector store, and a LangChain-based chat agent with summarization middleware and SQLite checkpointing. Emphasises reliability, configurability, and operational readiness for enterprise workloads.

---

## Project Highlights

- Document ingestion and chunking (RecursiveCharacterTextSplitter)
- Gemini embeddings via langchain-google-genai
- In-memory vector search persisted to data/vectorstore.json
- LangChain chat agent with SummarizationMiddleware for long-running sessions
- Durable checkpoints (data/checkpoints.sqlite) and thread/session registry (data/thread_id.txt)
- Explicit RAG tool (search_knowledge_base) used only for user-uploaded docs

---

## Key Features

- Fast local indexing of .md/.txt files with overlap-aware chunking
- Embedding pipeline that can be rebuilt or loaded from disk
- Multi-thread chat with session summaries and transcripts
- Environment-driven tuning (summarization frequency, message retention)
- Clear separation: ingestion (ingest.py), RAG core (rag.py), agent cli (agent.py)

---

## Architecture Overview

- CLI front-end: agent.py — interactive chat, session management, tools
- Ingestion: ingest.py -> rag.ingest_documents builds the vector store
- RAG core: rag.py — loaders, splitter, embeddings, vector store factory
- Persistence: data/vectorstore.json (vector store), data/checkpoints.sqlite (chat checkpoints), data/thread_id.txt (thread metadata)

---

## Tech Stack

- Python 3.10+
- LangChain / langchain-core
- langchain-google-genai (Gemini embeddings + chat)
- langgraph (checkpointing)
- dotenv, numpy

---

## Quickstart (Windows)

1. Clone repo and enter project root
   git clone https://github.com/junyuan-project/ai-agent-2026.git
   cd "D:\AI Agent"
2. Create virtualenv and install
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
3. Configure credentials and tune env vars
   - Add GOOGLE_API_KEY or set GOOGLE_APPLICATION_CREDENTIALS per Google GenAI docs
   - Optional: AGENT_SUMMARIZE_AT_MESSAGES, AGENT_KEEP_MESSAGES
4. Add docs: place .md/.txt files into the documents\ directory
5. Build the vector index
   python ingest.py
6. Run the agent
   python agent.py

Commands inside agent: new, threads, thread <id>, quit

---

## Design Notes & Tradeoffs

- Uses JSON-persisted InMemoryVectorStore for reproducible startups; swap for FAISS or SQLite for scale.
- SummarizationMiddleware reduces token usage; configurable to trade cost for context.
- RAG tool is opt-in to avoid over-reliance on private docs for general facts.

---

## Skills Demonstrated

LLM integration, embedding pipelines, vector search, chunking strategies, stateful CLI tooling, durable checkpointing, environment-driven config, and operational thinking (persistence, summaries, transcripts).

---

## Future Improvements

- Add automated tests and CI
- Replace JSON vector dump with FAISS/SQLite-backed store for larger corpora
- Add a web UI and REST API for broader demos
- Containerise + CI/CD and deploy to cloud

---

## 👨‍💻 Author

JY Wong

Software Engineer specializing in backend development, API integration, and healthcare interoperability solutions.

LinkedIn: https://www.linkedin.com/in/jun-yuan-wong-66b094233/

GitHub: https://github.com/junyuan-project

---

## 📄 License

This project is developed for educational and portfolio purposes.
