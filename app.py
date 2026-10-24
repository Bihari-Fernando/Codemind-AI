
import io
import zipfile
from pathlib import PurePosixPath

import streamlit as st

st.set_page_config(
    page_title="CodeMind AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 CodeMind AI")
st.write("Upload a ZIP codebase and explore its source files.")

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".dart", ".go", ".rs", ".cpp",
    ".c", ".h", ".cs", ".php", ".rb",
    ".swift", ".kt", ".scala", ".sql",
    ".html", ".css", ".scss", ".json",
    ".yaml", ".yml", ".toml", ".md",
    ".txt", ".sh", ".xml",
}

IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", "dist", "build", ".next",
    "vendor", "coverage", ".idea", ".vscode",
}

MAX_ZIP_SIZE = 50 * 1024 * 1024
MAX_FILES = 3000
MAX_FILE_SIZE = 1024 * 1024
MAX_TOTAL_SIZE = 20 * 1024 * 1024


def is_safe_source(path):
    # ZIP paths normally use forward slashes.
    if "\\" in path or "\x00" in path:
        return False

    p = PurePosixPath(path)

    if p.is_absolute() or ".." in p.parts:
        return False

    if not p.parts:
        return False

    if any(part.startswith(".") for part in p.parts[:-1]):
        return False

    if any(part.lower() in IGNORED_DIRS for part in p.parts):
        return False

    name = p.name.lower()

    # Never index common secrets or credential files.
    if name == ".env" or name.startswith(".env."):
        return False

    if any(word in name for word in (
        "secret", "credential", "private_key"
    )):
        return False

    if name.endswith((".pem", ".key", ".p12", ".pfx")):
        return False

    return p.suffix.lower() in ALLOWED_EXTENSIONS


uploaded_zip = st.file_uploader(
    "Upload your project ZIP",
    type=["zip"],
)

if uploaded_zip:
    zip_bytes = uploaded_zip.getvalue()

    if len(zip_bytes) > MAX_ZIP_SIZE:
        st.error("ZIP exceeds the 50 MB upload limit.")
        st.stop()

    source_files = {}
    total_size = 0

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            members = archive.infolist()

            if len(members) > MAX_FILES:
                st.error("ZIP contains too many entries.")
                st.stop()

            for item in members:
                if item.is_dir():
                    continue

                if not is_safe_source(item.filename):
                    continue

                # Reject symbolic links and oversized files.
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    continue

                if item.file_size > MAX_FILE_SIZE:
                    continue

                total_size += item.file_size

                if total_size > MAX_TOTAL_SIZE:
                    st.error("Selected source files exceed 20 MB.")
                    st.stop()

                content = archive.read(item)

                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                source_files[item.filename] = text

    except zipfile.BadZipFile:
        st.error("Invalid ZIP file. Please upload a valid ZIP.")
        st.stop()

    st.success(f"Found {len(source_files)} source files.")

    if source_files:
        st.subheader("Project files")

        st.write("These files are ready for the next indexing step.")

        selected_file = st.selectbox(
            "Choose a file to preview",
            sorted(source_files.keys()),
        )

        st.caption(f"File: {selected_file}")

        st.code(
            source_files[selected_file][:12000],
            language="text",
        )

        with st.expander("View all detected file paths"):
            for path in sorted(source_files.keys()):
                st.text(path)
    else:
        st.warning(
            "No supported source files found. "
            "Check that your ZIP contains code files."
        )
else:
    st.info("Upload a ZIP file to begin.")
