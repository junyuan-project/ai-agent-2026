# Sample knowledge base

This is example content for your RAG agent. Replace or add your own `.txt` and `.md` files in the `documents/` folder.

## Project overview

The AI Agent project uses LangGraph and Google Gemini. It supports interactive chat and retrieval-augmented generation (RAG).

## How to refresh the knowledge base

1. Put `.txt` or `.md` files in the `documents/` folder.
2. Run `py ingest.py` to rebuild the vector index.
3. Ask questions in `py agent.py`; the agent will search your documents when needed.

## Tips

- Use clear headings and short sections for better retrieval.
- Re-run ingest after adding or editing files.
