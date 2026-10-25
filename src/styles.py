"""Custom CSS layered on top of the dark theme in .streamlit/config.toml."""
import streamlit as st

CSS = """
<style>
:root {
  --cm-bg: #0E1014;
  --cm-surface: #15181E;
  --cm-surface-2: #1B1F27;
  --cm-border: #262A33;
  --cm-border-strong: #343945;
  --cm-text: #E4E6EC;
  --cm-muted: #9AA0AD;
  --cm-faint: #6D7380;
  --cm-accent: #7C83F7;
  --cm-accent-soft: rgba(124, 131, 247, 0.12);
  --cm-green: #4CC38A;
  --cm-amber: #E7B04B;
  --cm-red: #EF6B6B;
  --cm-blue: #6BA8F7;
}

/* Layout ------------------------------------------------------------- */
[data-testid="stMainBlockContainer"] {
  max-width: 1040px;
  padding: 2.25rem 1.5rem 7rem;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { border-right: 1px solid var(--cm-border); }
[data-testid="stSidebarContent"] { padding-top: 0.25rem; }
[data-testid="stSidebarUserContent"] { padding: 1rem 1rem 1.5rem; }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 7rem; }
}

h1, h2, h3 { letter-spacing: -0.015em; }
code { font-size: 0.86em; }
[data-testid="stMarkdownContainer"] p > code,
[data-testid="stMarkdownContainer"] li > code,
[data-testid="stMarkdownContainer"] td > code {
  color: #C7CBFA; background: rgba(124,131,247,0.10); border-radius: 5px; padding: 0.1em 0.35em;
}
[data-testid="stElementContainer"]:has(.cm-user-msg) { display: none; }

/* Brand -------------------------------------------------------------- */
.cm-brand { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.35rem; }
.cm-logo {
  width: 30px; height: 30px; border-radius: 8px; flex: none;
  background: var(--cm-accent-soft); border: 1px solid rgba(124,131,247,.35);
  display: grid; place-items: center; color: var(--cm-accent);
  font: 600 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.cm-brand-name { font-weight: 600; font-size: 1.02rem; color: var(--cm-text); }
.cm-brand-sub { color: var(--cm-muted); font-size: 0.82rem; line-height: 1.45; margin: 0 0 0.9rem; }

/* Small text helpers ------------------------------------------------- */
.cm-section-label {
  color: var(--cm-faint); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase; margin: 0.4rem 0 0.45rem;
}
.cm-muted { color: var(--cm-muted); font-size: 0.86rem; line-height: 1.5; }
.cm-label { color: var(--cm-muted); font-size: 0.8rem; font-weight: 600; margin: 0.9rem 0 0.25rem; }

/* Status pill -------------------------------------------------------- */
.cm-pill {
  display: inline-flex; align-items: center; gap: 0.4rem; white-space: nowrap;
  padding: 0.16rem 0.6rem; border-radius: 999px; font-size: 0.76rem; font-weight: 600;
  border: 1px solid var(--cm-border-strong); color: var(--cm-muted); background: var(--cm-surface-2);
}
.cm-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.cm-pill.ready { color: var(--cm-green); border-color: rgba(76,195,138,.35); background: rgba(76,195,138,.08); }
.cm-pill.indexing { color: var(--cm-blue); border-color: rgba(107,168,247,.35); background: rgba(107,168,247,.08); }
.cm-pill.error { color: var(--cm-red); border-color: rgba(239,107,107,.35); background: rgba(239,107,107,.08); }
@media (prefers-reduced-motion: no-preference) {
  .cm-pill.indexing .dot { animation: cm-pulse 1.2s ease-in-out infinite; }
}
@keyframes cm-pulse { 50% { opacity: .3; } }

/* Sidebar project block --------------------------------------------- */
.cm-project-name {
  font-weight: 600; font-size: 0.9rem; color: var(--cm-text);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 0.35rem;
}
.cm-project-meta { color: var(--cm-muted); font-size: 0.8rem; margin-top: 0.4rem; }
.cm-recent { list-style: none; padding: 0 !important; margin: 0 !important; }
.cm-recent li {
  margin: 0 !important;
  color: var(--cm-muted); font-size: 0.82rem; line-height: 1.4; padding: 0.38rem 0.5rem;
  border-radius: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cm-recent li:hover { background: var(--cm-surface-2); color: var(--cm-text); }
.cm-sidebar-foot { color: var(--cm-faint); font-size: 0.74rem; line-height: 1.5; margin-top: 1.25rem; }

/* Hero (upload screen) ---------------------------------------------- */
.cm-hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.cm-eyebrow {
  display: inline-block; color: var(--cm-accent); font-size: 0.78rem; font-weight: 600;
  border: 1px solid rgba(124,131,247,.3); background: var(--cm-accent-soft);
  padding: 0.2rem 0.7rem; border-radius: 999px; margin-bottom: 1.1rem;
}
.cm-hero h1 {
  font-size: clamp(2rem, 4.5vw, 2.75rem); font-weight: 650; line-height: 1.12;
  margin: 0 0 0.75rem; padding: 0; color: var(--cm-text);
}
.cm-hero p { color: var(--cm-muted); font-size: 1.05rem; margin: 0 auto; max-width: 34rem; line-height: 1.55; }

.cm-limits { display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem; margin-top: 0.15rem; }
.cm-limits span { color: var(--cm-muted); font-size: 0.8rem; }
.cm-limits b { color: var(--cm-text); font-weight: 600; }

.cm-examples { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.cm-chip {
  color: var(--cm-muted); font-size: 0.84rem; padding: 0.38rem 0.75rem; border-radius: 999px;
  border: 1px solid var(--cm-border); background: var(--cm-surface);
}

/* Uploader ----------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed var(--cm-border-strong); background: var(--cm-surface); padding: 1.6rem 1.25rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cm-accent); }

/* Cards (keyed containers) ------------------------------------------ */
.st-key-upload_card, .st-key-overview_card {
  background: var(--cm-surface);
}

/* Stat tiles --------------------------------------------------------- */
.cm-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.6rem; margin: 0.2rem 0 0.4rem; }
.cm-stat { background: var(--cm-surface-2); border: 1px solid var(--cm-border); border-radius: 10px; padding: 0.7rem 0.85rem; min-width: 0; }
.cm-stat .k { color: var(--cm-faint); font-size: 0.72rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.cm-stat .v { color: var(--cm-text); font-size: 1.15rem; font-weight: 600; margin-top: 0.2rem;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cm-stat .s { color: var(--cm-muted); font-size: 0.76rem; margin-top: 0.1rem; }
@media (max-width: 760px) { .cm-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.cm-ws-head { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.15rem; }
.cm-ws-head h2 { font-size: 1.35rem; font-weight: 600; margin: 0; padding: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 100%; }

.cm-folder { color: var(--cm-faint); font-size: 0.76rem; font-weight: 600; margin: 0.6rem 0 0.15rem; }
.cm-file { color: var(--cm-text); font: 0.82rem/1.65 ui-monospace, SFMono-Regular, Menlo, monospace; padding-left: 0.6rem; }

/* Chat --------------------------------------------------------------- */
[data-testid="stChatMessage"] { background: transparent; padding: 0.6rem 0.25rem; gap: 0.85rem; }
[data-testid="stChatMessage"]:has(.cm-user-msg) {
  background: var(--cm-surface); border: 1px solid var(--cm-border); border-radius: 12px; padding: 0.7rem 0.9rem;
}
[data-testid="stBottomBlockContainer"] { max-width: 1040px; padding-bottom: 1.25rem; }
[data-testid="stChatInput"] textarea { font-size: 0.95rem; }

.st-key-suggestions button > div { justify-content: flex-start; width: 100%; }
.st-key-suggestions button p { text-align: left; }
.st-key-suggestions button {
  justify-content: flex-start; text-align: left; height: 100%; padding: 0.6rem 0.9rem;
  background: var(--cm-surface); border-color: var(--cm-border); color: var(--cm-muted);
}
.st-key-suggestions button:hover { border-color: var(--cm-accent); color: var(--cm-text); }

/* Answer sections ---------------------------------------------------- */
[class*="st-key-short_"] {
  background: var(--cm-accent-soft); border: 1px solid rgba(124,131,247,.28);
  border-radius: 10px; padding: 0.75rem 0.95rem 0.15rem;
}
[class*="st-key-limits_"] {
  background: rgba(231,176,75,.06); border: 1px solid rgba(231,176,75,.25);
  border-radius: 10px; padding: 0.7rem 0.95rem 0.1rem;
}
[class*="st-key-security_"] {
  background: rgba(239,107,107,.05); border: 1px solid rgba(239,107,107,.22);
  border-radius: 10px; padding: 0.7rem 0.95rem 0.1rem;
}
.cm-callout-label { font-size: 0.74rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; margin-bottom: 0.3rem; }
.cm-callout-label.short { color: var(--cm-accent); }
.cm-callout-label.limits { color: var(--cm-amber); }
.cm-callout-label.security { color: var(--cm-red); }

.cm-sources { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.15rem 0 0.35rem; }
.cm-source {
  display: inline-flex; gap: 0.45rem; align-items: baseline; max-width: 100%;
  border: 1px solid var(--cm-border); background: var(--cm-surface); border-radius: 8px; padding: 0.28rem 0.6rem;
}
.cm-source .p { color: var(--cm-text); font: 0.8rem ui-monospace, SFMono-Regular, Menlo, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cm-source .l { color: var(--cm-faint); font-size: 0.75rem; white-space: nowrap; }

[data-testid="stExpander"] details { border-color: var(--cm-border); background: var(--cm-surface); }
[data-testid="stExpander"] summary:hover { color: var(--cm-text); }

/* Buttons ------------------------------------------------------------ */
.stButton button, .stDownloadButton button { font-weight: 550; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
