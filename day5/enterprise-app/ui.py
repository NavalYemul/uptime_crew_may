import os
import time
from typing import Any

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
LOCAL_PROMPTS = [
    "What is Orbit?",
    "Which teams use Orbit?",
    "What are the support hours?",
]
WEB_PROMPTS = [
    "Who is the CEO of OpenAI?",
    "What happened in AI this week?",
    "What is the latest NVIDIA market cap?",
]
WELCOME_MESSAGE = (
    "Ask about the local demo docs or try something current to trigger the "
    "`web_search` tool."
)


def fetch_status() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask_backend(query: str) -> dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"query": query},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def initialize_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("pending_prompt", "")
    st.session_state.setdefault("latest_response", None)
    st.session_state.setdefault("latest_error", None)
    st.session_state.setdefault("activity_log", [])


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f3ede2;
            --panel: rgba(255, 250, 241, 0.9);
            --panel-strong: rgba(255, 255, 255, 0.92);
            --ink: #1f2a2a;
            --muted: #60706a;
            --accent: #0f766e;
            --accent-soft: #daf3ed;
            --warm: #c96f3b;
            --warm-soft: #f7dfd2;
            --border: rgba(31, 42, 42, 0.1);
            --shadow: 0 20px 50px rgba(62, 40, 15, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(201, 111, 59, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 22%),
                linear-gradient(180deg, #f8f2e8 0%, var(--bg) 48%, #eee3d4 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }

        .shell {
            background: linear-gradient(180deg, rgba(255, 250, 241, 0.8), rgba(255, 255, 255, 0.65));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }

        .hero {
            background: linear-gradient(135deg, rgba(255, 250, 241, 0.96), rgba(245, 255, 253, 0.88));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.2rem 1.3rem 1rem;
            margin-bottom: 1rem;
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--warm);
            margin-bottom: 0.5rem;
        }

        .hero-title {
            font-size: 2.3rem;
            line-height: 1;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 52rem;
            margin-bottom: 0.85rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.74rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--border);
            color: var(--ink);
            font-size: 0.84rem;
        }

        .rail-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .rail-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.6rem;
            font-weight: 700;
        }

        .rail-value {
            font-size: 1.5rem;
            font-weight: 800;
            line-height: 1.05;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }

        .rail-copy {
            color: var(--muted);
            font-size: 0.93rem;
        }

        .chat-stage {
            color: var(--muted);
            font-size: 0.9rem;
            margin: 0.5rem 0 0;
        }

        .badge {
            display: inline-block;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            background: var(--accent-soft);
            color: var(--accent);
        }

        .badge-warm {
            background: var(--warm-soft);
            color: var(--warm);
        }

        .resource-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem;
            margin-bottom: 0.75rem;
        }

        .resource-title {
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.3rem;
        }

        .resource-meta {
            font-size: 0.82rem;
            color: var(--muted);
            margin-bottom: 0.45rem;
        }

        .activity-item {
            border-left: 2px solid rgba(15, 118, 110, 0.22);
            padding-left: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .activity-label {
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.15rem;
        }

        .activity-copy {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .stChatMessage {
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid var(--border);
            border-radius: 20px;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(15, 118, 110, 0.18);
            background: white;
        }

        @media (max-width: 900px) {
            .hero-title { font-size: 1.9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(status: dict[str, Any] | None) -> None:
    data_count = len(status["data_files"]) if status else 0
    model_name = status["model"] if status else "Unavailable"
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">Conversational Demo</div>
            <div class="hero-title">Enterprise RAG Agent Chat</div>
            <div class="hero-copy">
                Use this like a product demo: ask an internal question to stay grounded in local docs,
                then ask something current to show the agent calling <code>web_search</code>.
            </div>
            <div class="chip-row">
                <span class="chip">Backend: {BACKEND_URL}</span>
                <span class="chip">Model: {model_name}</span>
                <span class="chip">Local files: {data_count}</span>
                <span class="chip">Tool fallback: web_search</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def queue_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def clear_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.latest_response = None
    st.session_state.latest_error = None
    st.session_state.activity_log = []
    st.session_state.pending_prompt = ""


def append_activity(label: str, copy: str) -> None:
    st.session_state.activity_log.insert(0, {"label": label, "copy": copy})
    st.session_state.activity_log = st.session_state.activity_log[:8]


def build_activity_from_response(query: str, data: dict[str, Any]) -> list[dict[str, str]]:
    entries = [
        {
            "label": "User Question",
            "copy": query,
        },
        {
            "label": "Local Retrieval",
            "copy": f"Retrieved {len(data.get('local_documents', []))} local chunk(s).",
        },
        {
            "label": "Agent Decision",
            "copy": f"Strategy: {data['tool_strategy']}. Source: {data['source']}.",
        },
    ]

    for item in data.get("tool_calls", []):
        entries.append(
            {
                "label": "Tool Call",
                "copy": f"{item['tool']} ran with query '{item['query']}'.",
            }
        )

    entries.append(
        {
            "label": "Answer Ready",
            "copy": "Response rendered with supporting evidence and resources.",
        }
    )
    return entries


def run_query(query: str) -> None:
    st.session_state.latest_error = None
    with st.status("Running the agent...", expanded=True) as status_box:
        status_box.write("Queued the user message.")
        time.sleep(0.2)
        status_box.write("Checking local knowledge first.")
        time.sleep(0.2)
        data = ask_backend(query)
        status_box.write(
            f"Retrieved {len(data.get('local_documents', []))} local chunk(s)."
        )
        if data.get("tool_calls"):
            for tool_call in data["tool_calls"]:
                status_box.write(
                    f"Called `{tool_call['tool']}` with query `{tool_call['query']}`."
                )
        else:
            status_box.write("No web tool was needed for this turn.")
        status_box.write("Rendering answer, resources, and activity feed.")
        status_box.update(label="Agent run complete", state="complete")

    st.session_state.latest_response = data
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "data": data,
        }
    )
    for entry in reversed(build_activity_from_response(query, data)):
        append_activity(entry["label"], entry["copy"])


def render_prompt_bar() -> None:
    st.caption("Jump to a local-doc question or a web-fallback demo:")
    st.markdown("**Local-first prompts**")
    for column, prompt in zip(st.columns(3), LOCAL_PROMPTS, strict=True):
        with column:
            st.button(
                prompt,
                key=f"prompt-{prompt}",
                use_container_width=True,
                on_click=queue_prompt,
                args=(prompt,),
            )
    st.markdown("**Web fallback prompts**")
    for column, prompt in zip(st.columns(3), WEB_PROMPTS, strict=True):
        with column:
            st.button(
                prompt,
                key=f"prompt-{prompt}",
                use_container_width=True,
                on_click=queue_prompt,
                args=(prompt,),
            )


def render_chat_history() -> None:
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(WELCOME_MESSAGE)
            st.markdown(
                "`What is Orbit?` is a good local-doc opener. `Who is the CEO of OpenAI?` is a good tool-call demo."
            )
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "data" in message:
                data = message["data"]
                badges = [
                    f"<span class='badge'>{data['source']}</span>",
                    f"<span class='badge'>{data['tool_strategy']}</span>",
                ]
                if data.get("tool_calls"):
                    badges.append("<span class='badge badge-warm'>web_search used</span>")
                st.markdown("".join(badges), unsafe_allow_html=True)

                with st.expander("Why this answer?", expanded=False):
                    st.write(f"Local chunks: `{len(data.get('local_documents', []))}`")
                    st.write(f"Tool calls: `{len(data.get('tool_calls', []))}`")
                    if data.get("tool_calls"):
                        for item in data["tool_calls"]:
                            st.write(
                                f"- `{item['tool']}` with query `{item['query']}` ({item['mode']})"
                            )
                    evidence_tab, web_tab, raw_tab = st.tabs(
                        ["Local evidence", "Web evidence", "Raw response"]
                    )

                    with evidence_tab:
                        local_docs = data.get("local_documents", [])
                        if not local_docs:
                            st.caption("No local evidence was used for this answer.")
                        else:
                            for item in local_docs[:4]:
                                with st.container(border=True):
                                    st.markdown(
                                        f"**{item['source']}**  •  confidence `{item['confidence']}`"
                                    )
                                    st.write(item["content"])

                    with web_tab:
                        web_results = data.get("web_results", [])
                        if not web_results:
                            st.caption("No web results were needed for this answer.")
                        else:
                            for item in web_results[:4]:
                                with st.container(border=True):
                                    st.markdown(
                                        f"**[{item['title']}]({item['url']})**"
                                    )
                                    if item.get("snippet"):
                                        st.caption(item["snippet"])

                    with raw_tab:
                        st.json(data)


def render_sidebar_panel(title: str, body: str, value: str | None = None) -> None:
    value_markup = f"<div class='rail-value'>{value}</div>" if value else ""
    st.markdown(
        f"""
        <div class="rail-card">
            <div class="rail-title">{title}</div>
            {value_markup}
            <div class="rail-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_activity_feed() -> None:
    st.markdown('<div class="rail-card">', unsafe_allow_html=True)
    st.markdown('<div class="rail-title">Live Activity</div>', unsafe_allow_html=True)
    if not st.session_state.activity_log:
        st.caption("The activity feed will fill in as the agent handles questions.")
    else:
        for item in st.session_state.activity_log[:5]:
            st.markdown(
                f"""
                <div class="activity-item">
                    <div class="activity-label">{item['label']}</div>
                    <div class="activity-copy">{item['copy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_debug_panel() -> None:
    data = st.session_state.latest_response
    st.markdown('<div class="rail-card">', unsafe_allow_html=True)
    st.markdown('<div class="rail-title">Latest Turn</div>', unsafe_allow_html=True)
    if not data:
        st.caption("No completed turn yet.")
    else:
        st.write(f"Source: `{data['source']}`")
        st.write(f"Strategy: `{data['tool_strategy']}`")
        st.write(f"Local chunks: `{len(data.get('local_documents', []))}`")
        st.write(f"Tool calls: `{len(data.get('tool_calls', []))}`")
        st.write(f"Cache hit: `{data['cache_hit']}`")
    st.markdown("</div>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Enterprise RAG Agent Chat",
    page_icon=":speech_balloon:",
    layout="wide",
)
initialize_state()
inject_styles()

status = fetch_status()

st.markdown('<div class="shell">', unsafe_allow_html=True)
render_hero(status)

main_col, rail_col = st.columns([1.75, 0.65], gap="large")

with main_col:
    render_prompt_bar()
    render_chat_history()

with rail_col:
    if status:
        render_sidebar_panel(
            "Backend",
            f"Connected to `{BACKEND_URL}` with `{len(status['data_files'])}` local file(s).",
            "Connected",
        )
        render_sidebar_panel(
            "Model",
            "Used for query rewriting, answer generation, and tool-use decisions.",
            status["model"],
        )
    else:
        render_sidebar_panel(
            "Backend",
            "The status endpoint is not reachable. Start FastAPI first, then refresh.",
            "Offline",
        )
    render_activity_feed()
    render_debug_panel()
    st.button("Clear Conversation", use_container_width=True, on_click=clear_chat)

st.markdown("</div>", unsafe_allow_html=True)

prompt = st.chat_input("Ask a question about the local docs or something current...")
queued_prompt = st.session_state.pending_prompt
active_prompt = prompt or queued_prompt

if active_prompt:
    st.session_state.pending_prompt = ""
    try:
        run_query(active_prompt)
        st.rerun()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        st.session_state.latest_error = detail
        append_activity("Run Failed", detail)
        st.rerun()
    except requests.RequestException as exc:
        st.session_state.latest_error = f"Request failed: {exc}"
        append_activity("Run Failed", str(exc))
        st.rerun()

if st.session_state.latest_error:
    st.error(st.session_state.latest_error)
