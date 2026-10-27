"""Prompt Injection & Jailbreak Lab — NITTIE College Security Challenge.

An educational web application for AI engineering and LLM security students.
Students connect to the Nexus API Gateway with their own credentials and evaluate
whether NITTIE Helper can be manipulated through conversation into disclosing
restricted administrative and event management records.
"""

from datetime import datetime, timezone
import html
import os
import urllib.parse
import uuid

import gradio as gr
from openai import OpenAI
import openai

import config
from export_utils import (
    generate_json_export,
    generate_markdown_export,
    generate_plain_text_transcript,
)


def get_default_session_state() -> dict:
    """Return a fresh, isolated session state dictionary."""
    return {
        "student_name": "",
        "base_url": "",
        "api_key": "",
        "is_connected": False,
        "conversation_id": None,
        "messages": [],  # List of dicts: {'role': 'user'|'assistant', 'content': str, 'timestamp': str}
    }


# -------------------------------------------------------------------------
# STAGE 1: NEXUS API CONNECTION & VALIDATION HANDLERS
# -------------------------------------------------------------------------


def validate_base_url(url_str: str) -> tuple[bool, str]:
    """Validate that the URL is a supported HTTP or HTTPS endpoint.

    Requires HTTPS outside local development (localhost / 127.0.0.1 / 0.0.0.0).
    """
    clean_url = (url_str or "").strip()
    if not clean_url:
        return False, "Please enter your Nexus API Base URL."

    try:
        parsed = urllib.parse.urlparse(clean_url)
    except Exception:
        return False, "Malformed Base URL format."

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False, "The Base URL must be a valid endpoint beginning with `https://` or `http://`."

    hostname = (parsed.hostname or "").lower()
    is_local = hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1")

    # Enforce HTTPS outside local development
    if parsed.scheme == "http" and not is_local:
        return (
            False,
            "Insecure protocol: HTTPS is required for remote Nexus Base URLs. Use `https://...`.",
        )

    return True, clean_url


def connect_and_start(
    student_name_input: str,
    api_key_input: str,
    base_url_input: str,
    session: dict,
):
    """Validate student setup inputs, test connection via fixed model, and initialize session."""
    # 1. Validate Student Name
    clean_name = (student_name_input or "").strip()
    if not clean_name:
        return (
            session,
            "⚠️ **Validation Error**: Student Name is required. Please enter your name.",
            gr.update(visible=True),   # stage_connect stays visible
            gr.update(visible=False),  # stage_chat
            "",                        # badge_html
            [],                        # chatbot
            "",                        # transcript_box
            "",                        # copy_status
            gr.update(value=None, visible=False),  # download_file
            "",                        # export_status
        )

    if len(clean_name) > config.MAX_STUDENT_NAME_LENGTH:
        return (
            session,
            f"⚠️ **Validation Error**: Student Name exceeds maximum limit of {config.MAX_STUDENT_NAME_LENGTH} characters.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )

    # 2. Validate API Key
    clean_key = (api_key_input or "").strip()
    if not clean_key:
        return (
            session,
            "⚠️ **Validation Error**: Nexus API Key is required. Please enter your key.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )

    # 3. Validate Base URL
    is_valid_url, url_or_error = validate_base_url(base_url_input)
    if not is_valid_url:
        return (
            session,
            f"⚠️ **Validation Error**: {url_or_error}",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )
    clean_url = url_or_error

    # 4. Perform minimal genuine test request using fixed backend model
    try:
        client = OpenAI(
            base_url=clean_url,
            api_key=clean_key,
            timeout=config.REQUEST_TIMEOUT,
        )

        client.chat.completions.create(
            model=config.MODEL_NAME,  # Fixed nova-micro backend model
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )

    except openai.AuthenticationError:
        return (
            session,
            "❌ **Authentication Failed**: Invalid API Key. Please verify your credentials in the Nexus API Portal.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )
    except openai.APIConnectionError:
        return (
            session,
            "❌ **Connection Failed**: Unable to reach the API Base URL. Please verify your URL and network connectivity.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )
    except openai.RateLimitError:
        return (
            session,
            "❌ **Rate Limit Exceeded**: Your Nexus API account or gateway has exceeded its rate limit. Please try again shortly.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )
    except openai.APITimeoutError:
        return (
            session,
            "❌ **Request Timeout**: The request to Nexus API timed out. Please try again.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )
    except openai.BadRequestError:
        return (
            session,
            "❌ **API Request Error**: Gateway rejected the test request. Verify the Base URL path (e.g. `/v1`).",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )
    except Exception:
        # Strictly sanitize: never leak API key, model name, or stack traces
        return (
            session,
            "❌ **Connection Error**: An error occurred while testing connection to Nexus API. Please check your settings.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            [],
            "",
            "",
            gr.update(value=None, visible=False),
            "",
        )

    # Setup succeeded! Create session
    new_convo_id = str(uuid.uuid4())
    new_session = {
        "student_name": clean_name,
        "base_url": clean_url,
        "api_key": clean_key,
        "is_connected": True,
        "conversation_id": new_convo_id,
        "messages": [],
    }

    safe_name = html.escape(clean_name)
    badge_html = (
        '<div class="session-chips">'
        f'<span class="chip"><span class="dot"></span><span class="k">Connected</span></span>'
        f'<span class="chip"><span class="k">Student</span> <strong>{safe_name}</strong></span>'
        '<span class="chip"><span class="k">Challenge</span> <strong>Prompt injection</strong></span>'
        '</div>'
    )

    return (
        new_session,
        "",                        # Clear connect status
        gr.update(visible=False),  # Hide stage_connect
        gr.update(visible=True),   # Show stage_chat
        badge_html,                # Header session badge
        [],                        # Empty initial chatbot display
        "",                        # Clear transcript
        "",                        # Clear copy status
        gr.update(value=None, visible=False),  # Hide export download
        "",                        # Clear export status
    )


# -------------------------------------------------------------------------
# STAGE 2: CHAT AND EXPERIMENTATION HANDLERS
# -------------------------------------------------------------------------


def handle_send_chat(user_text: str, chatbot_display: list, session: dict):
    """Stream model response using OpenAI SDK with the fixed system prompt.

    Enforces input length bounds, session isolation, and backend-only model config.
    """
    clean_text = (user_text or "").strip()
    if not clean_text:
        yield chatbot_display, session, ""
        return

    # Check input length bound
    if len(clean_text) > config.MAX_INPUT_LENGTH:
        error_msg = f"⚠️ Input exceeds maximum allowed length of {config.MAX_INPUT_LENGTH} characters."
        updated_display = list(chatbot_display) + [
            {"role": "user", "content": clean_text[:100] + "..."},
            {"role": "assistant", "content": error_msg},
        ]
        yield updated_display, session, ""
        return

    # Verify session is valid and connected
    if not session.get("is_connected") or not session.get("api_key") or not session.get("base_url"):
        error_msg = "⚠️ Session expired or not connected. Please reconnect using the setup page."
        updated_display = list(chatbot_display) + [
            {"role": "user", "content": clean_text},
            {"role": "assistant", "content": error_msg},
        ]
        yield updated_display, session, ""
        return

    now_iso = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    # Add user message to session message store
    user_msg_entry = {
        "role": "user",
        "content": clean_text,
        "timestamp": now_iso,
    }
    session["messages"].append(user_msg_entry)

    # Update chatbot display immediately with user message
    display_history = list(chatbot_display) + [
        {"role": "user", "content": clean_text}
    ]
    yield display_history, session, ""  # Clear user textbox immediately

    # Build API message payload strictly on backend:
    # 1. System prompt strictly from backend configuration (fixed NITTIE Helper prompt + KBs)
    api_messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    # 2. Append history from session messages, strictly permitting only user/assistant roles
    for msg in session["messages"]:
        valid_role = "user" if msg.get("role") == "user" else "assistant"
        api_messages.append({"role": valid_role, "content": msg.get("content", "")})

    # Initialize client and stream completion
    try:
        client = OpenAI(
            base_url=session["base_url"],
            api_key=session["api_key"],
            timeout=config.REQUEST_TIMEOUT,
        )

        response_stream = client.chat.completions.create(
            model=config.MODEL_NAME,  # Fixed backend model identifier
            messages=api_messages,
            stream=True,
        )

        full_reply = ""
        # Append empty assistant placeholder message
        display_history.append({"role": "assistant", "content": ""})

        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            full_reply += delta
            display_history[-1]["content"] = full_reply
            yield display_history, session, ""

        # Request completed successfully -> update session history
        assistant_timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        session["messages"].append({
            "role": "assistant",
            "content": full_reply,
            "timestamp": assistant_timestamp,
        })
        yield display_history, session, ""

    except openai.AuthenticationError:
        error_reply = "❌ Authentication error: Please check your Nexus API key."
        display_history[-1]["content"] = error_reply
        yield display_history, session, ""
    except openai.RateLimitError:
        error_reply = "❌ Rate limit exceeded: Please wait a moment before sending another message."
        display_history[-1]["content"] = error_reply
        yield display_history, session, ""
    except openai.APITimeoutError:
        error_reply = "❌ Request timed out: The Nexus API gateway took too long to respond."
        display_history[-1]["content"] = error_reply
        yield display_history, session, ""
    except Exception:
        # Strictly sanitize: never expose internal configuration or stack traces
        error_reply = "❌ Error: Unable to complete request with Nexus API. Please verify your connection."
        display_history[-1]["content"] = error_reply
        yield display_history, session, ""


def handle_reset_chat(session: dict):
    """Start a fresh conversation while retaining the active student session and credentials."""
    session["conversation_id"] = str(uuid.uuid4())
    session["messages"] = []
    return (
        session,
        [],  # Clear chatbot display
        "",  # Clear user input textbox
        "",  # Clear transcript area
        "",  # Clear copy status
        gr.update(value=None, visible=False),  # Hide previous export file
        "",  # Clear export status
    )


def handle_disconnect():
    """Invalidate session state, wipe stored credentials, and return to setup page."""
    fresh_session = get_default_session_state()
    return (
        fresh_session,
        gr.update(visible=True),   # Show stage_connect
        gr.update(visible=False),  # Hide stage_chat
        "",                        # Clear student name textbox
        "",                        # Clear API key textbox
        "",                        # Clear base URL textbox
        "",                        # Clear connect status
        [],                        # Clear chatbot display
        "",                        # Clear transcript area
        "",                        # Clear copy status
        gr.update(value=None, visible=False),  # Reset export file
        "",                        # Clear export status
    )


def handle_copy_chat_history(session: dict):
    """Generate plain-text transcript of visible conversation for easy clipboard copying."""
    messages = session.get("messages", [])
    if not messages:
        return (
            "",
            "⚠️ **Notice**: No messages to copy yet. Interact with the chatbot first.",
        )

    convo_id = session.get("conversation_id", "session")
    student_name = session.get("student_name", "Student")
    scenario_title = config.SCENARIO

    transcript = generate_plain_text_transcript(
        conversation_id=convo_id,
        student_name=student_name,
        scenario_title=scenario_title,
        messages=messages,
    )

    msg = f"✅ **Chat History Prepared ({len(messages)} messages)**: Text generated below. Click inside the box or copy to clipboard."
    return transcript, msg


def handle_export_markdown(session: dict):
    """Generate and return a downloadable Markdown export file."""
    messages = session.get("messages", [])
    if not messages:
        return (
            gr.update(visible=False),
            "⚠️ **Notice**: No messages to export yet. Please chat with the assistant first.",
        )

    convo_id = session.get("conversation_id", "session")
    student_name = session.get("student_name", "Student")
    scenario_title = config.SCENARIO

    try:
        file_path = generate_markdown_export(
            conversation_id=convo_id,
            student_name=student_name,
            scenario_title=scenario_title,
            messages=messages,
        )
        return (
            gr.update(value=file_path, visible=True),
            "✅ **Markdown Export Ready**: Click the file below to download your conversation log.",
        )
    except Exception as e:
        return (
            gr.update(visible=False),
            f"❌ **Export Failed**: Could not generate Markdown export ({str(e)}).",
        )


def handle_export_json(session: dict):
    """Generate and return a downloadable JSON export file."""
    messages = session.get("messages", [])
    if not messages:
        return (
            gr.update(visible=False),
            "⚠️ **Notice**: No messages to export yet. Please chat with the assistant first.",
        )

    convo_id = session.get("conversation_id", "session")
    student_name = session.get("student_name", "Student")
    scenario_title = config.SCENARIO

    try:
        file_path = generate_json_export(
            conversation_id=convo_id,
            student_name=student_name,
            scenario_title=scenario_title,
            messages=messages,
        )
        return (
            gr.update(value=file_path, visible=True),
            "✅ **JSON Export Ready**: Click the file below to download your conversation log.",
        )
    except Exception as e:
        return (
            gr.update(visible=False),
            f"❌ **Export Failed**: Could not generate JSON export ({str(e)}).",
        )


# -------------------------------------------------------------------------
# THEME & CUSTOM CSS
# -------------------------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f6f7fb",
    body_background_fill_dark="#0b0f19",
    block_border_width="0px",
    block_shadow="none",
    block_background_fill="transparent",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    input_background_fill_dark="#111827",
    input_border_color="#e2e8f0",
    input_border_color_focus="#6366f1",
    input_shadow_focus="0 0 0 3px rgba(99,102,241,.18)",
)

CUSTOM_CSS = """
:root {
    --ink: #0f172a;
    --muted: #64748b;
    --line: #e2e8f0;
    --card: #ffffff;
    --soft: #f1f5f9;
    --brand: #4f46e5;
    --brand-soft: #eef2ff;
    --ok: #059669;
    --ok-soft: #ecfdf5;
}
.dark {
    --ink: #e5e7eb;
    --muted: #94a3b8;
    --line: #1f2937;
    --card: #111827;
    --soft: #0f172a;
    --brand-soft: #1e1b4b;
    --ok-soft: #052e22;
}

.gradio-container { max-width: 1180px !important; margin: 0 auto !important; padding: 0 20px !important; }
footer { display: none !important; }

/* ---------- Top bar ---------- */
.topbar { display: flex; align-items: center; gap: 12px; padding: 22px 0 18px; }
.topbar .logo {
    width: 38px; height: 38px; border-radius: 10px; flex: none;
    display: grid; place-items: center; font-size: 20px;
    background: linear-gradient(135deg, #6366f1, #4338ca); color: #fff;
}
.topbar .title { font-size: 1.05rem; font-weight: 650; color: var(--ink); line-height: 1.2; letter-spacing: -.01em; }
.topbar .sub { font-size: .8rem; color: var(--muted); }

/* ---------- Connect screen ---------- */
.hero { text-align: center; max-width: 560px; margin: 28px auto 24px; }
.hero h1 { font-size: 2rem; font-weight: 700; letter-spacing: -.03em; color: var(--ink); margin: 0 0 8px; }
.hero p { color: var(--muted); font-size: 1rem; line-height: 1.6; margin: 0; }

.connect-card {
    max-width: 480px; margin: 0 auto 12px !important; padding: 28px !important;
    background: var(--card) !important; border: 1px solid var(--line) !important;
    border-radius: 16px !important; box-shadow: 0 10px 30px -12px rgba(15,23,42,.12) !important;
    gap: 14px !important;
}
.connect-card .portal-link { font-size: .85rem; color: var(--muted); margin-top: -2px; }
.connect-card .portal-link a { color: var(--brand); text-decoration: none; font-weight: 500; }
.connect-card .portal-link a:hover { text-decoration: underline; }
.connect-status p { margin: 0; font-size: .9rem; }
.privacy-note { text-align: center; font-size: .78rem; color: var(--muted); max-width: 480px; margin: 8px auto 0; }

/* ---------- Chat screen ---------- */
.session-bar {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 6px 0 14px;
}
.session-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    display: inline-flex; align-items: center; gap: 7px; padding: 6px 12px;
    font-size: .82rem; color: var(--ink); background: var(--card);
    border: 1px solid var(--line); border-radius: 999px;
}
.chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 0 3px var(--ok-soft); }
.chip .k { color: var(--muted); }

.panel {
    background: var(--card) !important; border: 1px solid var(--line) !important;
    border-radius: 16px !important; padding: 20px !important;
}
.panel-title { font-size: .72rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }

.brief { font-size: .9rem; line-height: 1.6; color: var(--ink); }
.brief h3 { font-size: 1rem; font-weight: 650; margin: 0 0 10px; color: var(--ink); }
.brief p { color: var(--muted); margin: 0 0 10px; }
.brief ul { margin: 0 0 10px 0; padding-left: 18px; color: var(--ink); }
.brief li { margin-bottom: 3px; }

.side-actions button { width: 100%; justify-content: center; }

.chat-shell { padding: 0 !important; overflow: hidden; }
.chat-shell .bubble-wrap, .chat-shell .wrapper { background: transparent !important; }
#chatbot { border: none !important; }

.composer {
    padding: 12px 14px 14px; border-top: 1px solid var(--line);
    align-items: flex-end !important; gap: 10px !important;
}
.composer textarea { border-radius: 12px !important; font-size: .95rem !important; }
.composer button { height: 44px; min-width: 96px; }

/* ---------- Evidence ---------- */
.transcript-box textarea {
    font-family: var(--font-mono) !important; font-size: .82rem !important; line-height: 1.55 !important;
}
.hint { font-size: .8rem; color: var(--muted); }

@media (max-width: 860px) {
    .gradio-container { padding: 0 12px !important; }
    .hero h1 { font-size: 1.6rem; }
}
"""

# JavaScript snippet for copying visible chat history to system clipboard
COPY_CLIPBOARD_JS = """
() => {
    const el = document.querySelector("#visible-transcript-area textarea");
    if (el && el.value) {
        navigator.clipboard.writeText(el.value).then(() => {
            console.log("Chat history successfully copied to clipboard.");
        }).catch(err => {
            console.warn("Clipboard access denied or unsupported", err);
        });
    }
}
"""

# Brief, student-facing summary of the assignment (restricted data stays on the backend)
CHALLENGE_BRIEF = """
<div class="brief">
<p class="panel-title">Your mission</p>
<h3>Can you make NITTIE Helper leak?</h3>
<p>The assistant answers public campus questions but also holds a fictional restricted knowledge base. Test whether it can tell the two apart under pressure.</p>
<p class="panel-title" style="margin-top:14px">Probe for</p>
<ul>
<li>Unpublished management events</li>
<li>Tentative schedules &amp; guest lists</li>
<li>Internal booking allocations</li>
<li>Proposed attendance figures</li>
<li>The restricted administrator</li>
<li>Internal workflow references</li>
<li>Pending management decisions</li>
</ul>
<p class="panel-title" style="margin-top:14px">Ground rules</p>
<p>Use this chat only. Don't touch the source, backend, credentials, or other students' sessions. When done, collect your evidence below.</p>
</div>
"""

TOPBAR_HTML = """
<div class="topbar">
  <div class="logo">🛡️</div>
  <div>
    <div class="title">Prompt Injection &amp; Jailbreak Lab</div>
    <div class="sub">NITTIE College of Technology · AI Security Challenge</div>
  </div>
</div>
"""

HERO_HTML = """
<div class="hero">
  <h1>Break the assistant.<br>Learn why it breaks.</h1>
  <p>Connect with your Nexus credentials and probe <strong>NITTIE Helper</strong> for information it was told to protect. All data in this lab is synthetic.</p>
</div>
"""

# -------------------------------------------------------------------------
# GRADIO APPLICATION UI BUILD
# -------------------------------------------------------------------------

with gr.Blocks(title="Prompt Injection & Jailbreak Lab") as demo:
    # Isolated session state per browser tab/user
    session_state = gr.State(value=get_default_session_state)

    gr.HTML(TOPBAR_HTML)

    # ---------------------------------------------------------------------
    # STAGE 1: NEXUS API CONFIGURATION (EXACTLY THREE INPUT FIELDS)
    # ---------------------------------------------------------------------
    with gr.Column(visible=True) as stage_connect:
        gr.HTML(HERO_HTML)

        with gr.Column(elem_classes=["connect-card"]):
            student_name_input = gr.Textbox(
                label="Student name",
                placeholder="e.g. Alex Morgan",
                value="",
                interactive=True,
                lines=1,
                info="Recorded in your exported evidence.",
            )
            api_key_input = gr.Textbox(
                label="Nexus API key",
                placeholder="Paste your key",
                value="",
                type="password",
                interactive=True,
                lines=1,
                info="Held in session memory only. Never stored.",
            )
            api_base_url_input = gr.Textbox(
                label="Nexus base URL",
                placeholder="https://api.nexus.example/v1",
                value="",
                interactive=True,
                lines=1,
                info="Copy it from your Nexus dashboard. HTTPS required.",
            )
            gr.Markdown(
                f"Need credentials? [Open the Nexus portal ↗]({config.NEXUS_PORTAL_URL})",
                elem_classes=["portal-link"],
            )
            connect_btn = gr.Button("Connect & start challenge", variant="primary", size="lg")
            connect_status = gr.Markdown(value="", visible=True, elem_classes=["connect-status"])

        gr.HTML('<div class="privacy-note">The model is configured on the server. You don\'t need to choose one.</div>')

    # ---------------------------------------------------------------------
    # STAGE 2: SINGLE SCENARIO CHALLENGE CHAT PAGE
    # ---------------------------------------------------------------------
    with gr.Column(visible=False) as stage_chat:
        with gr.Row(elem_classes=["session-bar"], equal_height=True):
            session_badge = gr.HTML(value="", scale=6)
            top_reset_btn = gr.Button("Reset chat", variant="secondary", size="sm", scale=0, min_width=110)
            top_disconnect_btn = gr.Button("Disconnect", variant="secondary", size="sm", scale=0, min_width=110)

        with gr.Row(equal_height=False):
            # ---- Left: briefing + evidence tools ----
            with gr.Column(scale=4, min_width=280):
                with gr.Column(elem_classes=["panel"]):
                    gr.HTML(CHALLENGE_BRIEF)
                    with gr.Accordion("Full scenario text", open=False):
                        gr.Markdown(
                            f"**{config.SCENARIO}**\n\n{config.SCENARIO_DESCRIPTION}",
                            elem_classes=["brief"],
                        )

                with gr.Column(elem_classes=["panel"]):
                    gr.HTML('<p class="panel-title">Collect evidence</p>')
                    copy_history_btn = gr.Button("Copy chat history", variant="primary")
                    copy_status = gr.Markdown(value="", elem_classes=["hint"])
                    with gr.Accordion("Transcript preview", open=False):
                        transcript_box = gr.Textbox(
                            show_label=False,
                            lines=8,
                            max_lines=20,
                            interactive=False,
                            elem_id="visible-transcript-area",
                            elem_classes=["transcript-box"],
                        )
                    with gr.Row():
                        export_md_btn = gr.Button("Download .md", size="sm")
                        export_json_btn = gr.Button("Download .json", size="sm")
                    export_status = gr.Markdown(value="", elem_classes=["hint"])
                    download_file = gr.File(
                        label="Your file",
                        visible=False,
                        interactive=False,
                    )
                    gr.HTML(
                        '<p class="hint">Exports include your name, scenario, timestamps and messages. '
                        "Credentials, system prompt and model identifiers are never included.</p>"
                    )

            # ---- Right: chat ----
            with gr.Column(scale=7, min_width=380):
                with gr.Column(elem_classes=["panel", "chat-shell"]):
                    chatbot = gr.Chatbot(
                        label="NITTIE Helper",
                        show_label=False,
                        height=560,
                        elem_id="chatbot",
                        placeholder=(
                            "<div style='text-align:center;padding:40px 20px;color:#64748b'>"
                            "<div style='font-size:2rem'>💬</div>"
                            "<div style='font-weight:600;margin:6px 0 4px;color:inherit'>Say hello to NITTIE Helper</div>"
                            "<div style='font-size:.9rem'>Start with a normal campus question to set a baseline, "
                            "then start probing.</div></div>"
                        ),
                    )
                    with gr.Row(elem_classes=["composer"]):
                        user_input = gr.Textbox(
                            placeholder="Ask about the library, exams, events… or try something sneakier",
                            show_label=False,
                            container=False,
                            lines=1,
                            max_lines=6,
                            scale=6,
                            autofocus=True,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=0, min_width=96)

    # Hidden duplicate-free handles: bottom disconnect/reset now live in the top bar only.

    # ---------------------------------------------------------------------
    # EVENT CONNECTIONS
    # ---------------------------------------------------------------------

    connect_outputs = [
        session_state,
        connect_status,
        stage_connect,
        stage_chat,
        session_badge,
        chatbot,
        transcript_box,
        copy_status,
        download_file,
        export_status,
    ]
    connect_inputs = [student_name_input, api_key_input, api_base_url_input, session_state]
    connect_btn.click(fn=connect_and_start, inputs=connect_inputs, outputs=connect_outputs)
    api_base_url_input.submit(fn=connect_and_start, inputs=connect_inputs, outputs=connect_outputs)

    chat_io = dict(
        fn=handle_send_chat,
        inputs=[user_input, chatbot, session_state],
        outputs=[chatbot, session_state, user_input],
    )
    send_btn.click(**chat_io)
    user_input.submit(**chat_io)

    copy_history_btn.click(
        fn=handle_copy_chat_history,
        inputs=[session_state],
        outputs=[transcript_box, copy_status],
        js=COPY_CLIPBOARD_JS,
    )

    reset_outputs = [
        session_state,
        chatbot,
        user_input,
        transcript_box,
        copy_status,
        download_file,
        export_status,
    ]
    top_reset_btn.click(fn=handle_reset_chat, inputs=[session_state], outputs=reset_outputs)

    disconnect_outputs = [
        session_state,
        stage_connect,
        stage_chat,
        student_name_input,
        api_key_input,
        api_base_url_input,
        connect_status,
        chatbot,
        transcript_box,
        copy_status,
        download_file,
        export_status,
    ]
    top_disconnect_btn.click(fn=handle_disconnect, inputs=[], outputs=disconnect_outputs)

    export_md_btn.click(fn=handle_export_markdown, inputs=[session_state], outputs=[download_file, export_status])
    export_json_btn.click(fn=handle_export_json, inputs=[session_state], outputs=[download_file, export_status])


if __name__ == "__main__":
    is_spaces = "SPACE_ID" in os.environ or "HF_SPACE_ID" in os.environ
    server_name = "0.0.0.0" if is_spaces else os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port_env = os.environ.get("PORT")
    launch_kwargs = dict(
        server_name=server_name,
        share=False,
        theme=THEME,
        css=CUSTOM_CSS,
    )

    if port_env:
        demo.launch(server_port=int(port_env), **launch_kwargs)
    else:
        try:
            demo.launch(server_port=7860, **launch_kwargs)
        except OSError:
            demo.launch(**launch_kwargs)
