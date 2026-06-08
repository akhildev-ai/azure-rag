from pathlib import Path

from pypdf import PdfReader


def load_document(file_path: Path) -> list[dict]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix in {".txt", ".md"}:
        return _load_text(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(file_path: Path) -> list[dict]:
    reader = PdfReader(str(file_path))
    documents: list[dict] = []

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(
            {
                "content": text,
                "metadata": {
                    "source_file": file_path.name,
                    "page_number": i,
                },
            }
        )

    return documents


def _load_text(file_path: Path) -> list[dict]:
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [
        {
            "content": text,
            "metadata": {
                "source_file": file_path.name,
                "page_number": 1,
            },
        }
    ]
