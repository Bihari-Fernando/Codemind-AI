"""Streamlit screens for CodeMind AI. All data shown here comes from the backend; nothing is mocked."""
import hashlib
import html
import zipfile
from dataclasses import asdict

import streamlit as st

from . import backend
from .presentation import (
    FriendlyError, count_zip_files, explain_error, group_by_folder, human_size,
    language_for, parse_answer,
)

EXAMPLE_QUESTIONS = [
    "Explain this project in simple terms.",
    "Where is authentication handled?",
    "How does the application process user input?",
    "Which files are responsible for the main features?",
    "What limitations can you identify from the available code?",
]

PREVIEW_CHARS = 8000  # same limit as the original preview

STATUS_LABELS = {
    "not_indexed": "Not indexed",
    "indexing": "Indexing",
    "ready": "Ready",
    "error": "Error",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "project": None,          # dict with name, size, hash, files, total_entries, expected_chunks
        "history": [],            # list of chat turns for the current project (this session only)
        "index_error": None,      # dict(FriendlyError) from the last failed indexing run
        "index_requested": False, # "auto" or True when the next run should index
        "upload_error": None,     # dict(FriendlyError) shown on the upload screen
        "uploader_key": 0,        # bump to reset the file uploader widget
        "pending_question": None, # question chosen from a suggestion button
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_project() -> None:
    """Return to the upload screen. Indexed data stays in chroma_db and is reused for the same ZIP."""
    st.session_state.project = None
    st.session_state.history = []
    st.session_state.index_error = None
    st.session_state.index_requested = False
    st.session_state.upload_error = None
    st.session_state.pending_question = None
    st.session_state.uploader_key += 1


def new_conversation() -> None:
    st.session_state.history = []
    st.session_state.pending_question = None


def request_indexing() -> None:
    st.session_state.index_error = None
    st.session_state.index_requested = True


def ask_suggestion(question: str) -> None:
    st.session_state.pending_question = question


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------

def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pill(status: str) -> str:
    return f'<span class="cm-pill {status}"><span class="dot"></span>{STATUS_LABELS[status]}</span>'


def show_error(err: dict, troubleshooting: str = "") -> None:
    """Plain-language error with a next step, and technical details tucked away."""
    st.error(f"**{err['title']}**  \n{err['next_step']}", icon=":material/error:")
    if err.get("details") or troubleshooting:
        with st.expander("Troubleshooting details", icon=":material/build:"):
            if troubleshooting:
                st.markdown(troubleshooting)
            if err.get("details"):
                st.code(err["details"], language="text", wrap_lines=True)


def friendly(error: Exception, action: str) -> dict:
    return asdict(explain_error(error, action))


# ---------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------

def get_backend():
    """(client, db, error_dict). Uses the original cached get_clients()."""
    try:
        client, db = backend.get_clients()
        return client, db, None
    except Exception as error:  # missing key or Chroma failure
        return None, None, friendly(error, "start the AI services")


def get_collection(db, project):
    # Same name and settings as the original app, so existing indexes are reused.
    return db.get_or_create_collection(
        name=f"code_{project['hash']}",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def compute_status(indexed: int | None, expected: int) -> str:
    if st.session_state.index_requested:
        return "indexing"
    if st.session_state.index_error:
        return "error"
    if indexed is None:
        return "not_indexed"
    if expected and indexed >= expected:
        return "ready"
    if indexed > 0:
        return "error"  # partially indexed: a previous run stopped part-way
    return "not_indexed"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def load_project(uploaded, auto_index: bool) -> None:
    """Validate and read the ZIP with the original backend rules, then open the workspace."""
    st.session_state.upload_error = None
    zip_bytes = uploaded.getvalue()

    if len(zip_bytes) > backend.MAX_ZIP_SIZE:
        st.session_state.upload_error = {
            "title": f"This ZIP is {human_size(len(zip_bytes))}, which is over the 50 MB limit.",
            "next_step": "Remove folders such as node_modules, .venv, build output or large assets, then zip the project again.",
            "details": "",
        }
        return

    try:
        files = backend.read_zip(zip_bytes)
        total_entries = count_zip_files(zip_bytes)
    except zipfile.BadZipFile as error:
        st.session_state.upload_error = {
            "title": "This file couldn't be opened as a ZIP archive.",
            "next_step": "It may be corrupted or a different format (for example .rar or .7z). Create a standard .zip and upload it again.",
            "details": f"{type(error).__name__}: {error}",
        }
        return
    except ValueError as error:  # the backend's own limit messages
        st.session_state.upload_error = {
            "title": f"This project is too large to index: {error}",
            "next_step": "Zip only the source folders you want to ask about, leaving out dependencies, build output and data files.",
            "details": f"Limits: at most {backend.MAX_FILES:,} entries in the ZIP and "
                       f"{human_size(backend.MAX_TOTAL_SIZE)} of source files in total.",
        }
        return
    except (OSError, RuntimeError) as error:
        st.session_state.upload_error = {
            "title": "The ZIP couldn't be read.",
            "next_step": "If it is password-protected, create an unencrypted ZIP and upload it again.",
            "details": f"{type(error).__name__}: {error}",
        }
        return

    if not files:
        st.session_state.upload_error = {
            "title": "No supported source files were found in this ZIP.",
            "next_step": "Check that the ZIP contains your project's code (for example .py, .js, .java files) rather than only binaries or images.",
            "details": (
                f"The ZIP has {total_entries:,} files, but all were skipped. Files are skipped when they are in hidden or "
                "dependency folders (.git, node_modules, .venv, build, dist…), have an unsupported extension, are larger "
                "than 1 MB, are not UTF-8 text, or look like secrets (.env, keys, credentials)."
            ),
        }
        return

    st.session_state.project = {
        "name": uploaded.name,
        "size": len(zip_bytes),
        "hash": hashlib.sha256(zip_bytes).hexdigest()[:16],  # same as the original collection naming
        "files": files,
        "total_entries": total_entries,
        "expected_chunks": len(backend.make_chunks(files)),
    }
    st.session_state.history = []
    st.session_state.index_error = None
    # "auto" = requested from the upload screen; skipped if this ZIP is already fully indexed.
    st.session_state.index_requested = "auto" if auto_index else False


def render_upload_screen() -> None:
    st.markdown(
        """
        <div class="cm-hero">
          <div class="cm-eyebrow">AI codebase assistant</div>
          <h1>Understand any codebase.</h1>
          <p>Upload a project ZIP and ask questions about its code in plain language.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 6, 1])
    with center:
        with st.container(border=True, key="upload_card"):
            uploaded = st.file_uploader(
                "Project ZIP",
                type=["zip"],
                key=f"uploader_{st.session_state.uploader_key}",
                help="Upload a .zip of your project's source code.",
            )
            st.markdown(
                f"""
                <div class="cm-limits">
                  <span><b>.zip</b> up to {human_size(backend.MAX_ZIP_SIZE)}</span>
                  <span>Source files up to <b>{human_size(backend.MAX_FILE_SIZE)}</b> each</span>
                  <span><b>{human_size(backend.MAX_TOTAL_SIZE)}</b> of code in total</span>
                  <span>Up to <b>{backend.MAX_FILES:,}</b> entries</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("What gets indexed?", icon=":material/info:"):
                exts = " ".join(f"`{e}`" for e in sorted(backend.ALLOWED_EXTENSIONS))
                st.markdown(
                    f"**Included:** text source files with these extensions: {exts}\n\n"
                    "**Skipped automatically:** hidden folders and dependency or build folders "
                    f"({', '.join(sorted(backend.IGNORED_DIRS))}), `.env` files, keys and certificates "
                    "(`.pem`, `.key`, `.p12`, `.pfx`), files whose names contain *secret*, *credential* or "
                    "*private_key*, files over 1 MB, symbolic links, and non-UTF-8 files.\n\n"
                    "**Privacy:** indexing and questions send the included source code to the Google Gemini API."
                )

            c1, c2 = st.columns(2)
            scan_index = c1.button(
                "Scan and index project", type="primary", icon=":material/bolt:",
                disabled=uploaded is None, use_container_width=True,
            )
            scan_only = c2.button(
                "Scan files only", icon=":material/folder_open:",
                disabled=uploaded is None, use_container_width=True,
                help="Check which files are detected before using any API quota.",
            )

            if uploaded is not None and (scan_index or scan_only):
                with st.spinner("Reading the ZIP and detecting source files…"):
                    load_project(uploaded, auto_index=scan_index)
                if st.session_state.project:
                    st.rerun()

        if st.session_state.upload_error:
            show_error(st.session_state.upload_error)

        st.markdown('<div class="cm-section-label" style="margin-top:1.6rem">Questions you can ask once a project is indexed</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="cm-examples">' + "".join(f'<span class="cm-chip">{esc(q)}</span>' for q in EXAMPLE_QUESTIONS) + "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(status: str | None, indexed: int | None) -> None:
    project = st.session_state.project
    with st.sidebar:
        st.markdown(
            """
            <div class="cm-brand"><div class="cm-logo">&lt;/&gt;</div><div class="cm-brand-name">CodeMind AI</div></div>
            <p class="cm-brand-sub">Ask questions about any codebase in plain language.</p>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "New conversation", icon=":material/add_comment:", use_container_width=True,
            on_click=new_conversation, disabled=not st.session_state.history,
            help="Clear the questions and answers for this project. The index is kept.",
        )

        st.markdown('<div class="cm-section-label" style="margin-top:1.2rem">Project</div>', unsafe_allow_html=True)
        if project:
            files_n = len(project["files"])
            chunks_txt = f"{indexed:,} of {project['expected_chunks']:,} chunks indexed" if indexed is not None else "Index status unavailable"
            st.markdown(
                f'<div class="cm-project-name" title="{esc(project["name"])}">{esc(project["name"])}</div>'
                f"{pill(status)}"
                f'<div class="cm-project-meta">{files_n:,} source files · {chunks_txt}</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.button(
                "Replace ZIP", icon=":material/swap_horiz:", use_container_width=True,
                on_click=reset_project, disabled=status == "indexing",
                help="Go back to the upload screen. Indexes for previous ZIPs are kept on disk and reused.",
            )
        else:
            st.markdown('<div class="cm-muted">No project loaded yet. Upload a ZIP to begin.</div>', unsafe_allow_html=True)

        st.markdown('<div class="cm-section-label" style="margin-top:1.4rem">This session</div>', unsafe_allow_html=True)
        questions = [t["question"] for t in st.session_state.history]
        if questions:
            items = "".join(f'<li title="{esc(q)}">{esc(q)}</li>' for q in reversed(questions[-8:]))
            st.markdown(f'<ul class="cm-recent">{items}</ul>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cm-muted">Your questions will appear here.</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="cm-sidebar-foot">Conversations last until you refresh or close this tab.<br>'
            f"Models: {esc(backend.EMBED_MODEL)} · {esc(backend.CHAT_MODEL)}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Workspace: project overview + indexing
# ---------------------------------------------------------------------------

def run_indexing(client, db, collection, project) -> None:
    """Call the original index_codebase and only report success after it returns."""
    try:
        with st.status("Indexing project…", expanded=True) as box:
            st.write("Preparing code chunks…")
            chunks = backend.make_chunks(project["files"])
            if not chunks:
                raise ValueError("No non-empty code chunks found.")
            st.write(f"Creating embeddings for {len(chunks):,} chunks with {backend.EMBED_MODEL}…")
            backend.index_codebase(client, db, collection, chunks)
            box.update(label=f"Indexed {len(chunks):,} chunks", state="complete", expanded=False)
    except Exception as error:
        st.session_state.index_error = friendly(error, "index the project")
    finally:
        st.session_state.index_requested = False
    st.rerun()


def render_overview(project, status, indexed, backend_error) -> None:
    files = project["files"]
    st.markdown(
        f'<div class="cm-ws-head"><h2 title="{esc(project["name"])}">{esc(project["name"])}</h2>{pill(status)}</div>',
        unsafe_allow_html=True,
    )

    skipped = project["total_entries"] - len(files)
    chunk_value = f"{indexed:,}" if indexed is not None else "—"
    st.markdown(
        f"""
        <div class="cm-stats">
          <div class="cm-stat"><div class="k">Source files</div><div class="v">{len(files):,}</div>
            <div class="s">{skipped:,} other files skipped</div></div>
          <div class="cm-stat"><div class="k">Indexed chunks</div><div class="v">{chunk_value}</div>
            <div class="s">of {project['expected_chunks']:,} expected</div></div>
          <div class="cm-stat"><div class="k">Chunking</div><div class="v">{backend.CHUNK_LINES} lines</div>
            <div class="s">{backend.OVERLAP_LINES}-line overlap per chunk</div></div>
          <div class="cm-stat"><div class="k">ZIP size</div><div class="v">{human_size(project['size'])}</div>
            <div class="s">{project['total_entries']:,} files in archive</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if backend_error:
        show_error(backend_error, troubleshooting=(
            "1. In the project folder, open or create a file named `.env`.\n"
            "2. Add the line `GEMINI_API_KEY=your-key` (get a key at https://aistudio.google.com/apikey).\n"
            "3. Stop the app with Ctrl+C and run it again."
        ))
    elif status == "not_indexed":
        st.markdown(
            '<div class="cm-muted">Index the project so CodeMind AI can search it. This sends the detected source files to '
            "the Gemini API to create embeddings, and may take a minute for larger projects.</div>",
            unsafe_allow_html=True,
        )
        st.button("Index project", type="primary", icon=":material/bolt:", on_click=request_indexing)
    elif status == "error":
        if st.session_state.index_error:
            show_error(st.session_state.index_error)
        else:
            st.warning(
                f"Indexing stopped part-way: {indexed:,} of {project['expected_chunks']:,} chunks are stored. "
                "Re-index to finish before asking questions.",
                icon=":material/warning:",
            )
        st.button("Retry indexing", type="primary", icon=":material/refresh:", on_click=request_indexing)
    elif status == "ready":
        st.button(
            "Re-index", icon=":material/refresh:", on_click=request_indexing,
            help="Rebuild the embeddings for this ZIP. Only needed if indexing settings changed.",
        )

    c1, c2 = st.columns(2)
    with c1:
        with st.expander(f"Detected files ({len(files):,})", icon=":material/folder:"):
            with st.container(height=300 if len(files) > 12 else "content", border=False):
                parts = []
                for folder, names in group_by_folder(files).items():
                    parts.append(f'<div class="cm-folder">{esc(folder)}/</div>')
                    parts.extend(f'<div class="cm-file">{esc(n)}</div>' for n in names)
                st.markdown('<div style="padding-bottom:.5rem">' + "".join(parts) + "</div>", unsafe_allow_html=True)
    with c2:
        with st.expander("Preview a source file", icon=":material/code:"):
            path = st.selectbox("File", sorted(files), label_visibility="collapsed")
            content = files[path]
            st.code(content[:PREVIEW_CHARS], language=language_for(path))
            if len(content) > PREVIEW_CHARS:
                st.caption(f"Showing the first {PREVIEW_CHARS:,} of {len(content):,} characters.")


# ---------------------------------------------------------------------------
# Workspace: chat
# ---------------------------------------------------------------------------

def render_sources(matches, turn_id: int) -> None:
    st.markdown('<div class="cm-label">Relevant source files</div>', unsafe_allow_html=True)
    chips = "".join(
        f'<span class="cm-source" title="{esc(m["path"])}"><span class="p">{esc(m["path"])}</span>'
        f'<span class="l">lines {m["start_line"]}–{m["end_line"]}</span></span>'
        for m in matches
    )
    st.markdown(f'<div class="cm-sources">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<div class="cm-label">Code evidence</div>', unsafe_allow_html=True)
    for m in matches:
        with st.expander(f"{m['path']}  ·  lines {m['start_line']}–{m['end_line']}", icon=":material/description:"):
            st.code(m["text"], language=language_for(m["path"]))
    st.caption("These are the code sections retrieved for this question. The answer is based only on them.")


def callout(kind: str, label: str, body: str, turn_id: int) -> None:
    with st.container(key=f"{kind}_{turn_id}"):
        css = {"short": "short", "limits": "limits", "security": "security"}[kind]
        st.markdown(f'<div class="cm-callout-label {css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(body)


def render_answer(turn, turn_id: int) -> None:
    parsed = parse_answer(turn["answer"])
    if parsed.structured:
        if parsed.preamble:
            st.markdown(parsed.preamble)
        if "short" in parsed.sections:
            callout("short", "Short answer", parsed.sections["short"], turn_id)
        if "how" in parsed.sections:
            st.markdown('<div class="cm-label">How it works</div>', unsafe_allow_html=True)
            st.markdown(parsed.sections["how"])
        if "security" in parsed.sections:
            callout("security", "Security findings", parsed.sections["security"], turn_id)
            st.caption(
                "Confirmed vulnerabilities are supported by the retrieved code. Configuration warnings come from "
                "documented settings. Unverified risks need more files to confirm."
            )
        for title, body in parsed.extra:
            st.markdown(f'<div class="cm-label">{esc(title)}</div>', unsafe_allow_html=True)
            st.markdown(body)
        if "limitations" in parsed.sections:
            callout("limits", "Limitations", parsed.sections["limitations"], turn_id)
    else:
        st.markdown(turn["answer"])

    render_sources(turn["matches"], turn_id)


def user_message(text: str) -> None:
    with st.chat_message("user", avatar=":material/person:"):
        # Invisible marker so CSS can style user messages differently from answers.
        st.markdown('<span class="cm-user-msg"></span>', unsafe_allow_html=True)
        st.markdown(text)


def render_turn(turn, turn_id: int) -> None:
    user_message(turn["question"])
    with st.chat_message("assistant", avatar=":material/neurology:"):
        if turn["kind"] == "answer":
            render_answer(turn, turn_id)
        elif turn["kind"] == "empty":
            st.warning(
                "**No matching code was found for this question.**  \n"
                "Try naming a feature, file, function or screen, for example \"Where is the login form validated?\"",
                icon=":material/search_off:",
            )
        else:
            show_error(turn["error"])


def process_question(client, collection, question: str) -> dict:
    """Run the original search_code + answer_question and return a turn record."""
    turn = {"question": question, "kind": "answer", "answer": "", "matches": [], "error": None}
    user_message(question)
    with st.chat_message("assistant", avatar=":material/neurology:"):
        with st.status("Searching the codebase…", expanded=False) as box:
            try:
                matches = backend.search_code(client, collection, question)
                if not matches:
                    turn["kind"] = "empty"
                    box.update(label="No matching code found", state="error")
                    return turn
                turn["matches"] = matches
                box.update(label=f"Found {len(matches)} relevant code sections. Writing the answer…")
                turn["answer"] = backend.answer_question(client, matches, question)
                box.update(label="Done", state="complete")
            except Exception as error:
                turn["kind"] = "error"
                turn["error"] = friendly(error, "answer your question")
                box.update(label="Something went wrong", state="error")
    return turn


def render_chat(client, collection, status) -> None:
    ready = status == "ready" and client is not None
    st.markdown('<div class="cm-section-label" style="margin-top:1.8rem">Conversation</div>', unsafe_allow_html=True)

    history = st.session_state.history
    if not history:
        if ready:
            st.markdown('<div class="cm-muted" style="margin-bottom:.6rem">Ask anything about this project, or start with a suggestion:</div>',
                        unsafe_allow_html=True)
            with st.container(key="suggestions"):
                cols = st.columns(2)
                for i, q in enumerate(EXAMPLE_QUESTIONS):
                    cols[i % 2].button(q, key=f"suggest_{i}", on_click=ask_suggestion, args=(q,),
                                       use_container_width=True)
        else:
            st.info("Questions unlock once the project has been indexed.", icon=":material/lock:")

    for i, turn in enumerate(history):
        render_turn(turn, i)

    typed = st.chat_input(
        "Ask about this codebase…" if ready else "Index the project to start asking questions",
        disabled=not ready,
    )
    question = typed or st.session_state.pop("pending_question", None)
    if question and question.strip():
        if not ready:
            st.warning("Index the project before asking questions.", icon=":material/lock:")
            return
        turn = process_question(client, collection, question.strip())
        st.session_state.history.append(turn)
        st.rerun()


def render_workspace() -> None:
    project = st.session_state.project
    client, db, backend_error = get_backend()

    collection, indexed = None, None
    if db is not None:
        try:
            collection = get_collection(db, project)
            indexed = collection.count()
        except Exception as error:
            backend_error = friendly(error, "open the local index (chroma_db)")
            st.session_state.index_requested = False

    if backend_error:
        st.session_state.index_requested = False
    elif (st.session_state.index_requested == "auto" and indexed is not None
          and indexed >= project["expected_chunks"] > 0):
        st.session_state.index_requested = False  # existing index for this exact ZIP is reused
    status = compute_status(indexed, project["expected_chunks"])
    render_sidebar(status, indexed)

    with st.container(border=True, key="overview_card"):
        render_overview(project, status, indexed, backend_error)
        if status == "indexing" and collection is not None:
            run_indexing(client, db, collection, project)

    render_chat(client, collection, status)
