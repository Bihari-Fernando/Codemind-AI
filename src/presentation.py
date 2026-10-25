"""Pure helpers for presenting backend results. No Streamlit calls here, so they are easy to test."""
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

# ---------------------------------------------------------------------------
# Answer sections
# ---------------------------------------------------------------------------

KNOWN_SECTIONS = {
    "short answer": "short",
    "how it works": "how",
    "security findings": "security",
    "limitations": "limitations",
}

_HEADING = re.compile(r"^\s{0,3}#{1,4}\s+(.+?)\s*#*\s*$")
_BOLD_HEADING = re.compile(r"^\s{0,3}\*\*(.+?)\*\*:?\s*$")


@dataclass
class ParsedAnswer:
    preamble: str = ""
    sections: dict = field(default_factory=dict)  # key -> markdown body
    extra: list = field(default_factory=list)     # [(title, body)] for unknown headings
    structured: bool = False


def parse_answer(text: str) -> ParsedAnswer:
    """Split a markdown answer on the headings the prompt asks for.

    Only splits on headings that are really in the text (outside code fences).
    If no known heading is found the answer is returned unstructured, so the UI
    shows it exactly as generated instead of inventing sections.
    """
    result = ParsedAnswer()
    current_key, current_title, buf = None, None, []
    in_fence = False

    def flush():
        body = "\n".join(buf).strip()
        if current_key is None:
            result.preamble = body
        elif current_key == "_extra":
            if body or current_title:
                result.extra.append((current_title, body))
        elif body:
            prev = result.sections.get(current_key)
            result.sections[current_key] = f"{prev}\n\n{body}" if prev else body

    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        title = None
        if not in_fence:
            m = _HEADING.match(line) or _BOLD_HEADING.match(line)
            if m:
                title = m.group(1).strip().strip("*").strip().rstrip(":")
        if title is not None:
            key = KNOWN_SECTIONS.get(title.lower())
            # A bold line counts as a section only if it is one of ours; otherwise it's
            # ordinary emphasis (e.g. "**Confirmed vulnerabilities**" inside a section).
            if key or _HEADING.match(line):
                flush()
                buf = []
                current_key, current_title = (key, title) if key else ("_extra", title)
                continue
        buf.append(line)
    flush()

    result.structured = bool(result.sections)
    return result


# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "jsx", ".ts": "typescript", ".tsx": "tsx",
    ".java": "java", ".dart": "dart", ".go": "go", ".rs": "rust", ".cpp": "cpp", ".c": "c",
    ".h": "c", ".cs": "csharp", ".php": "php", ".rb": "ruby", ".swift": "swift",
    ".kt": "kotlin", ".sql": "sql", ".html": "html", ".css": "css", ".scss": "scss",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".md": "markdown",
    ".txt": "text", ".sh": "bash", ".xml": "xml",
}


def language_for(path: str) -> str:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "text")


def count_zip_files(zip_bytes: bytes) -> int:
    """Number of file entries (not folders) in the archive. Reads the index only."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return sum(1 for item in archive.infolist() if not item.is_dir())


def group_by_folder(paths) -> dict:
    """{"src/api": ["auth.py", ...], "(root)": [...]} sorted for display."""
    groups: dict = {}
    for path in sorted(paths):
        p = PurePosixPath(path)
        folder = str(p.parent) if str(p.parent) != "." else "(project root)"
        groups.setdefault(folder, []).append(p.name)
    return groups


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} MB"


# ---------------------------------------------------------------------------
# Errors in plain language
# ---------------------------------------------------------------------------

_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def redact(text: str) -> str:
    """Remove anything that looks like a Google API key before showing technical details."""
    return _KEY_PATTERN.sub("[redacted API key]", text)


@dataclass
class FriendlyError:
    title: str
    next_step: str
    details: str


def explain_error(error: Exception, action: str) -> FriendlyError:
    """Map an exception to a short message and a practical next step.

    `action` is what the user was doing, e.g. "index the project" or "answer your question".
    """
    raw = redact(f"{type(error).__name__}: {error}")
    # Include the original cause: answer_question wraps API errors in RuntimeError.
    cause = error.__cause__
    if cause is not None:
        raw += "\n\nCaused by: " + redact(f"{type(cause).__name__}: {cause}")
    low = raw.lower()

    if "gemini_api_key is missing" in low:
        return FriendlyError(
            "Your Gemini API key isn't set up.",
            "Add GEMINI_API_KEY=your-key to the .env file in the project folder, then restart the app.",
            raw,
        )
    if any(t in low for t in ("429", "resource_exhausted", "quota", "rate limit")):
        return FriendlyError(
            "The Gemini API rate limit was reached.",
            "Wait about a minute and try again. Free-tier keys allow a limited number of requests per minute.",
            raw,
        )
    if any(t in low for t in ("api key not valid", "api_key_invalid", "permission_denied", "401", "403", "unauthenticated")):
        return FriendlyError(
            "Gemini rejected the API key.",
            "Check that GEMINI_API_KEY in .env is correct and active in Google AI Studio, then restart the app.",
            raw,
        )
    if any(t in low for t in ("503", "unavailable", "500", "internal", "timeout", "deadline", "busy")):
        return FriendlyError(
            "The Gemini service is temporarily busy or unavailable.",
            "Wait a few moments and try again.",
            raw,
        )
    if any(t in low for t in ("not_found", "404", "is not found", "not supported")):
        return FriendlyError(
            "The configured Gemini model wasn't found.",
            "Check EMBED_MODEL and CHAT_MODEL in src/backend.py against the models available to your key.",
            raw,
        )
    if any(t in low for t in ("connect", "network", "ssl", "proxy", "getaddrinfo", "name resolution")):
        return FriendlyError(
            "CodeMind AI couldn't reach the Gemini API.",
            "Check your internet connection (and any VPN or proxy), then try again.",
            raw,
        )
    return FriendlyError(
        f"CodeMind AI couldn't {action}.",
        "Try again. If it keeps failing, open the technical details below.",
        raw,
    )
