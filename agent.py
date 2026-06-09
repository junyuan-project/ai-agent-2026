import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver

from rag import DOCUMENTS_DIR, load_vector_store, make_search_tool

load_dotenv()

DATA_DIR = Path("data")
CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"
THREAD_ID_FILE = DATA_DIR / "thread_id.txt"

SUMMARIZE_AT_MESSAGES = int(os.getenv("AGENT_SUMMARIZE_AT_MESSAGES", "30"))
KEEP_MESSAGES = int(os.getenv("AGENT_KEEP_MESSAGES", "16"))
THREAD_SUMMARY_MAX_MESSAGES = int(os.getenv("AGENT_THREAD_SUMMARY_MAX_MESSAGES", "40"))
THREAD_HISTORY_MAX_MESSAGES = int(os.getenv("AGENT_THREAD_HISTORY_MAX_MESSAGES", "50"))
USE_LLM_THREAD_SUMMARY = os.getenv("AGENT_THREAD_SUMMARY_LLM", "false").lower() in (
    "1",
    "true",
    "yes",
)

THREAD_HISTORY_REQUEST_RE = re.compile(
    r"\b("
    r"thread\s*history|thread\s*histories|thread\s*ids?|"
    r"list\s+(my\s+)?(chats?|threads?|sessions?)|show\s+(my\s+)?(chats?|threads?|sessions?)|"
    r"all\s+threads?|my\s+threads?|my\s+chats?|chat\s+sessions?|session\s+history|"
    r"past\s+(chats?|conversations?|sessions?)|previous\s+(chats?|conversations?)|"
    r"conversation\s+history|current\s+thread|active\s+thread"
    r")\b",
    re.IGNORECASE,
)

THREAD_HISTORY_EXACT = frozenset(
    {
        "threads",
        "thread history",
        "list my chats",
        "list chats",
        "my chats",
        "show my chats",
        "show chats",
    }
)

THREAD_INDEX_PROMPT = """Write a brief session summary (2-4 sentences) for a chat index.
Include the user's main goal, topics covered, and any conclusions. No labels or preamble.

Conversation:
{conversation}
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_current_time() -> str:
    """Return the current local time as a string."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_tools():
    tools = [get_current_time]
    vector_store = load_vector_store()
    if vector_store is not None:
        tools.append(make_search_tool(vector_store))
    return tools, vector_store is not None


def build_system_prompt(has_rag: bool) -> str:
    system_prompt = (
        "You are an expert, highly knowledgeable assistant. "
        "When answering questions, provide clear, accurate explanations. "
        "Use tools when they help answer the question accurately. "
        "This app stores multi-turn chats on disk with thread_id sessions. "
        "When the user asks about thread history, thread_id, chat sessions, "
        "past conversations, or the current thread, call list_chat_threads — "
        "do not claim you lack access to prior sessions."
    )
    if has_rag:
        system_prompt += (
            " Use search_knowledge_base ONLY when the user asks about their uploaded "
            "documents, notes, or private knowledge base. For general knowledge "
            "questions (e.g. standards, coding, science), answer directly from your "
            "own knowledge without requiring a document search. If you search the "
            "knowledge base and find nothing relevant, answer from general knowledge "
            "anyway when you can; do not refuse just because the knowledge base lacks "
            "the topic. Cite document sources when RAG results are used."
        )
    return system_prompt


def _empty_registry(active: str | None = None) -> dict:
    default = active or os.getenv("AGENT_THREAD_ID", "default")
    return {"active": default, "threads": {}}


def load_thread_registry() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not THREAD_ID_FILE.exists():
        registry = _empty_registry()
        save_thread_registry(registry)
        return registry

    raw = THREAD_ID_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        registry = _empty_registry()
        save_thread_registry(registry)
        return registry

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "threads" in data and "active" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Legacy: file contained only a single thread_id line.
    legacy_id = raw.splitlines()[0].strip()
    registry = _empty_registry(legacy_id)
    registry["threads"][legacy_id] = {
        "summary": "",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "message_count": 0,
    }
    save_thread_registry(registry)
    return registry


def save_thread_registry(registry: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    THREAD_ID_FILE.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_thread_entry(registry: dict, thread_id: str) -> None:
    threads = registry.setdefault("threads", {})
    if thread_id not in threads:
        now = _utc_now()
        threads[thread_id] = {
            "summary": "No messages yet.",
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }


def set_active_thread(registry: dict, thread_id: str) -> None:
    ensure_thread_entry(registry, thread_id)
    registry["active"] = thread_id
    save_thread_registry(registry)


def new_thread(registry: dict) -> str:
    thread_id = uuid.uuid4().hex[:12]
    ensure_thread_entry(registry, thread_id)
    set_active_thread(registry, thread_id)
    return thread_id


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def conversation_messages(agent, thread_id: str) -> list:
    snapshot = agent.get_state(thread_config(thread_id))
    if not snapshot or not snapshot.values:
        return []
    return snapshot.values.get("messages", [])


def _human_message_text(message: HumanMessage) -> str:
    content = message.content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def local_thread_summary(messages: list) -> str:
    """Cheap index blurb — no API call."""
    if not messages:
        return "No messages yet."

    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    parts = [f"{len(messages)} message(s), {len(human_msgs)} from user."]
    if human_msgs:
        first = _human_message_text(human_msgs[0])
        if len(first) > 100:
            first = first[:97] + "..."
        parts.append(f'First: "{first}"')
        if len(human_msgs) > 1:
            last = _human_message_text(human_msgs[-1])
            if len(last) > 100:
                last = last[:97] + "..."
            parts.append(f'Latest: "{last}"')
    return " ".join(parts)


def summarize_conversation_llm(messages: list) -> str:
    recent = messages[-THREAD_SUMMARY_MAX_MESSAGES:]
    text = get_buffer_string(recent)
    if len(text) > 12_000:
        text = text[-12_000:]

    response = llm.invoke(
        THREAD_INDEX_PROMPT.format(conversation=text),
    )
    return response.text.strip()


def summarize_conversation(messages: list) -> str:
    if not messages:
        return "No messages yet."
    if not USE_LLM_THREAD_SUMMARY:
        return local_thread_summary(messages)
    try:
        return summarize_conversation_llm(messages)
    except Exception:
        return local_thread_summary(messages)


def sync_thread_metadata(agent, registry: dict, thread_id: str) -> None:
    """Update thread index from checkpoint — local summary by default (no extra API)."""
    messages = conversation_messages(agent, thread_id)
    count = len(messages)
    ensure_thread_entry(registry, thread_id)
    entry = registry["threads"][thread_id]

    if count == entry.get("message_count") and entry.get("summary"):
        return

    entry["summary"] = summarize_conversation(messages)
    entry["message_count"] = count
    entry["updated_at"] = _utc_now()
    save_thread_registry(registry)


def sync_all_thread_metadata(agent, registry: dict) -> None:
    for tid in list(registry.get("threads", {})):
        sync_thread_metadata(agent, registry, tid)


def format_message_line(message) -> str | None:
    if isinstance(message, HumanMessage):
        content = message.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return f"    User: {content}"
    if isinstance(message, AIMessage):
        text = message.text or (
            message.content if isinstance(message.content, str) else str(message.content)
        )
        if not text:
            return None
        return f"    Assistant: {text}"
    if isinstance(message, ToolMessage):
        preview = str(message.content)
        if len(preview) > 400:
            preview = preview[:397] + "..."
        return f"    Tool ({message.name or 'tool'}): {preview}"
    return None


def format_transcript(messages: list, *, max_messages: int = THREAD_HISTORY_MAX_MESSAGES) -> str:
    if not messages:
        return "    (no messages)"
    shown = messages[-max_messages:] if max_messages > 0 else messages
    lines: list[str] = []
    if len(messages) > len(shown):
        lines.append(f"    ... ({len(messages) - len(shown)} earlier messages omitted)")
    for message in shown:
        line = format_message_line(message)
        if line:
            lines.append(line)
    return "\n".join(lines) if lines else "    (no messages)"


def build_thread_history_report(
    agent,
    registry: dict,
    active_thread_id: str,
    *,
    include_transcripts: bool = True,
) -> str:
    sync_all_thread_metadata(agent, registry)
    threads = registry.get("threads", {})
    if not threads:
        return (
            f"Current thread_id: {active_thread_id}\n\n"
            "No saved threads in data/thread_id.txt yet."
        )

    lines = [
        "=" * 60,
        f"CURRENT thread_id: {active_thread_id}",
        "=" * 60,
        "",
        f"All sessions ({len(threads)} thread(s)), newest first:",
        "",
    ]

    sorted_ids = sorted(
        threads,
        key=lambda t: threads[t].get("updated_at", ""),
        reverse=True,
    )

    for tid in sorted_ids:
        entry = threads[tid]
        is_active = tid == active_thread_id
        header = f"--- thread_id: {tid}"
        if is_active:
            header += "  [CURRENT]"
        lines.append(header)
        lines.append(f"  created:  {entry.get('created_at', '?')}")
        lines.append(f"  updated:  {entry.get('updated_at', '?')}")
        messages = conversation_messages(agent, tid)
        lines.append(f"  messages: {len(messages)}")
        lines.append("  summary:")
        for summary_line in (entry.get("summary") or "(no summary)").splitlines():
            lines.append(f"    {summary_line}")
        if include_transcripts:
            lines.append("  conversation:")
            lines.append(format_transcript(messages))
        lines.append("")

    return "\n".join(lines).rstrip()


def wants_thread_history(user_text: str) -> bool:
    lower = user_text.lower().strip()
    if lower in THREAD_HISTORY_EXACT:
        return True
    return THREAD_HISTORY_REQUEST_RE.search(user_text) is not None


def make_list_chat_threads_tool(agent_holder: dict, registry: dict, get_active_thread_id):
    def list_chat_threads() -> str:
        """List every saved chat thread_id, the current session, summaries, and transcripts.

        Use when the user asks about thread history, thread_id, chat sessions,
        past conversations, or what was discussed in earlier chats.
        """
        return build_thread_history_report(
            agent_holder["agent"],
            registry,
            get_active_thread_id(),
            include_transcripts=True,
        )

    return list_chat_threads


def create_chat_agent(checkpointer: SqliteSaver, tools, has_rag: bool):
    return create_agent(
        llm,
        tools=tools,
        system_prompt=build_system_prompt(has_rag),
        middleware=[
            SummarizationMiddleware(
                llm,
                trigger=("messages", SUMMARIZE_AT_MESSAGES),
                keep=("messages", KEEP_MESSAGES),
            ),
        ],
        checkpointer=checkpointer,
    )


def last_assistant_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.text
    return ""


def main():
    tools, has_rag = build_tools()
    registry = load_thread_registry()

    print("AI Agent — type your question.")
    print(
        "Commands: quit, exit, q | new | threads | thread <id>\n"
        "Ask naturally: show thread history, current thread id, list my chats\n"
    )
    if has_rag:
        print("RAG: knowledge base loaded.\n")
    else:
        print(
            f"RAG: no index yet. Add .txt/.md files to {DOCUMENTS_DIR}, "
            "then run: py ingest.py\n"
        )

    thread_id = registry.get("active") or os.getenv("AGENT_THREAD_ID", "default")
    ensure_thread_entry(registry, thread_id)
    save_thread_registry(registry)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        active_thread = [thread_id]
        agent_holder: dict = {"agent": None}
        list_threads_tool = make_list_chat_threads_tool(
            agent_holder,
            registry,
            lambda: active_thread[0],
        )
        agent = create_chat_agent(
            checkpointer,
            [*tools, list_threads_tool],
            has_rag,
        )
        agent_holder["agent"] = agent

        config = thread_config(active_thread[0])

        prior = len(conversation_messages(agent, active_thread[0]))
        if prior:
            print(
                f"Current thread_id: {active_thread[0]} "
                f"({prior} messages in memory)\n"
            )
        else:
            print(f"Current thread_id: {active_thread[0]} (new conversation)\n")

        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                sync_thread_metadata(agent, registry, active_thread[0])
                print("\nBye.")
                break

            if not user_text:
                continue

            lower = user_text.lower()
            if lower in {"quit", "exit", "q"}:
                sync_thread_metadata(agent, registry, active_thread[0])
                print("Bye.")
                break
            if lower == "threads" or wants_thread_history(user_text):
                report = build_thread_history_report(
                    agent, registry, active_thread[0]
                )
                print(f"\n{report}\n")
                continue
            if lower == "new":
                sync_thread_metadata(agent, registry, active_thread[0])
                active_thread[0] = new_thread(registry)
                config = thread_config(active_thread[0])
                print(f"Started new chat. thread_id={active_thread[0]}\n")
                continue
            if lower.startswith("thread "):
                new_id = user_text.split(maxsplit=1)[1].strip()
                if not new_id:
                    print("Usage: thread <id>\n")
                    continue
                sync_thread_metadata(agent, registry, active_thread[0])
                active_thread[0] = new_id
                set_active_thread(registry, active_thread[0])
                config = thread_config(active_thread[0])
                prior = len(conversation_messages(agent, active_thread[0]))
                entry = registry["threads"][active_thread[0]]
                print(
                    f"Switched to thread_id={active_thread[0]} "
                    f"({prior} messages in memory)\n"
                    f"  Summary: {entry.get('summary', '')}\n"
                )
                continue

            result = agent.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config,
            )
            reply = last_assistant_text(result["messages"])
            sync_thread_metadata(agent, registry, active_thread[0])
            print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
