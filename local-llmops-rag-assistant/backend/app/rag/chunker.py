def chunk_documents(
    documents: list[dict],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[dict]:
    chunks: list[dict] = []
    separators = ["\n\n", "\n", ". ", " "]

    for doc in documents:
        source_file = doc["metadata"].get("source_file", "unknown")
        page_number = doc["metadata"].get("page_number", 1)
        split_content = _recursive_split(doc["content"], chunk_size, separators)
        split_content = _apply_overlap(split_content, chunk_overlap)

        for idx, piece in enumerate(split_content):
            chunk_id = f"{source_file}:{page_number}:{idx}"
            chunks.append(
                {
                    "id": chunk_id,
                    "content": piece,
                    "metadata": {
                        "source_file": source_file,
                        "page_number": page_number,
                        "chunk_id": chunk_id,
                    },
                }
            )

    return chunks


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    parts = text.split(separator)
    merged: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            merged.append(current.strip())
            current = part
        else:
            merged.extend(_recursive_split(part, chunk_size, separators[1:]))
            current = ""

    if current:
        merged.append(current.strip())

    final_chunks: list[str] = []
    for part in merged:
        if len(part) <= chunk_size:
            final_chunks.append(part)
        else:
            final_chunks.extend(_recursive_split(part, chunk_size, separators[1:]))

    return [chunk for chunk in final_chunks if chunk]


def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or not chunks:
        return chunks

    combined: list[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            combined.append(chunk)
            continue

        prefix = chunks[idx - 1][-overlap_chars:]
        combined.append(f"{prefix} {chunk}".strip())
    return combined
