"""
Task 4 — Chunking, embedding và indexing.

Pipeline:
    data/standardized/**/*.md -> Document -> Chunk -> embedding -> ChromaDB.

Chunking là structure-aware thay vì cắt mù theo ký tự: Markdown do Task 3 sinh ra
giữ nguyên ranh giới Chương / Mục / Điều nên chunker tách theo heading trước, chỉ
recursive-split phần thân khi một mục dài hơn CHUNK_SIZE. Mỗi chunk được gắn thêm
một dòng breadcrumb ("<tên văn bản> > Chương ... > Điều ...") ngay trong content —
nhờ vậy một chunk nằm giữa Điều 105 vẫn mang đủ ngữ cảnh cho cả dense lẫn BM25.
Breadcrumb được tính vào ngân sách độ dài nên content không vượt CHUNK_SIZE.

ID chunk là `<đường dẫn md>::chunk-<index>` — ổn định giữa các lần chạy. Kết hợp
với upsert và bước dọn chunk mồ côi, chạy lại pipeline không tạo dữ liệu trùng.

embed_texts() là điểm dùng chung: Task 5 phải gọi đúng hàm này để embed query,
tránh lệch model hoặc dimension giữa corpus và query.

Chạy:
    python -m src.task4_chunking_indexing
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
CHROMA_DIR = ROOT_DIR / "chroma_db"

load_dotenv(ROOT_DIR / ".env")

# --- Tham số chunking ------------------------------------------------------
# Starter đề xuất 500/50. Nhóm nâng lên 1000/150 dựa trên phân bố độ dài thật của
# corpus: 421 khối "Điều" trong 4 văn bản pháp luật có median 902 ký tự, p25 488.
# Ở 1000 ký tự, quá nửa số Điều nằm trọn trong một chunk nên retrieval trả về cả
# điều luật thay vì một mảnh khoản mất đầu; ở 500 ký tự phần lớn Điều bị cắt làm
# 2–4 mảnh. Overlap 150 (15%) đủ bắc cầu một câu tiếng Việt bị cắt ngang.
# Bộ giá trị này phải được ghi lại trong bảng run information của
# group_project/evaluation/RESULT.md ở lần chạy evaluation.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "markdown_recursive"

# --- Embedding -------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 16

COLLECTION_NAME = "rag_documents"

# --- Hằng số nội bộ của chunker -------------------------------------------
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
RULE_PATTERN = re.compile(r"^\s*-{3,}\s*$")
URL_PATTERNS = [
    re.compile(r"^-\s+\*\*Nguồn công khai:\*\*\s*(\S+)", re.MULTILINE),
    re.compile(r"^-\s+\*\*Nguồn:\*\*\s*(\S+)", re.MULTILINE),
    re.compile(r"^-\s+\*\*Trang văn bản chính thức:\*\*\s*(\S+)", re.MULTILINE),
]
TITLE_PATTERN = re.compile(r"^#\s+(.*\S)\s*$", re.MULTILINE)

BREADCRUMB_SEPARATOR = " > "
MAX_HEADING_CHARS = 90
MAX_BREADCRUMB_CHARS = 220
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

# ChromaDB giới hạn số bản ghi mỗi lần upsert.
UPSERT_BATCH_SIZE = 2000

_embedding_model = None
_splitters: dict[tuple[int, int], object] = {}


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text bằng provider trong .env.

    Điểm dùng chung của Task 4 (corpus) và Task 5 (query): cùng model, cùng
    dimension, cùng cách normalize.
    """
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "sentence_transformers":
        vectors = _embed_sentence_transformers(texts)
    elif EMBEDDING_PROVIDER == "openai":
        vectors = _embed_openai(texts)
    elif EMBEDDING_PROVIDER == "gemini":
        vectors = _embed_gemini(texts)
    else:
        raise ValueError(
            f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER!r}. "
            "Dùng sentence_transformers | openai | gemini."
        )

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise RuntimeError(f"Embedding dimension không đồng nhất: {sorted(dimensions)}")
    return vectors


def _get_sentence_transformer():
    """Nạp model một lần cho cả tiến trình (tránh load lại ở mỗi query)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    model = _get_sentence_transformer()
    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        # Normalize để cosine distance của Chroma = 1 - dot product.
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
    )
    return [vector.tolist() for vector in vectors]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=texts[start : start + 64]
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google import genai

    client = genai.Client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=texts[start : start + 64]
        )
        vectors.extend(list(item.values) for item in response.embeddings)
    return vectors


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_documents() -> list[dict]:
    """Đọc Markdown trong data/standardized và trả về danh sách Document."""
    if not STANDARDIZED_DIR.is_dir():
        raise FileNotFoundError(
            f"Thiếu {STANDARDIZED_DIR}. Chạy `python -m src.task3_convert_markdown` trước."
        )

    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        relative = path.relative_to(STANDARDIZED_DIR)
        title_match = TITLE_PATTERN.search(content)
        documents.append(
            {
                "id": relative.as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": title_match.group(1).strip() if title_match else path.stem,
                    "doc_type": relative.parts[0] if len(relative.parts) > 1 else "legal",
                    # Header do Task 3 sinh ra giữ lại URL nguồn — dùng cho citation.
                    "url": _extract_url(content),
                },
            }
        )

    if not documents:
        raise FileNotFoundError(f"Không tìm thấy Markdown nào trong {STANDARDIZED_DIR}.")
    return documents


def _extract_url(content: str) -> str | None:
    """Lấy URL nguồn từ header metadata của Markdown chuẩn hóa."""
    header = content.split("\n---", 1)[0]
    for pattern in URL_PATTERNS:
        match = pattern.search(header)
        if match:
            return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------
def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id ổn định, chunk_index và breadcrumb."""
    chunks: list[dict] = []
    for document in documents:
        index = 0
        for heading_path, body in _split_sections(document["content"]):
            breadcrumb = _breadcrumb(heading_path)
            for text in _split_body(body, breadcrumb):
                content = f"{breadcrumb}\n\n{text}" if breadcrumb else text
                chunks.append(
                    {
                        "id": f"{document['id']}::chunk-{index}",
                        "content": content,
                        "metadata": {
                            **document["metadata"],
                            "chunk_index": index,
                            "doc_id": document["id"],
                            "section": heading_path[-1] if heading_path else "",
                            "section_path": breadcrumb,
                        },
                    }
                )
                index += 1
    return chunks


def _split_sections(text: str) -> list[tuple[list[str], str]]:
    """Tách text theo heading Markdown, trả về (đường dẫn heading, thân đoạn)."""
    sections: list[tuple[list[str], str]] = []
    path: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(line for line in buffer if not RULE_PATTERN.match(line)).strip()
        if body:
            sections.append(([title for _, title in path], body))
        buffer.clear()

    for line in text.splitlines():
        match = HEADING_PATTERN.match(line)
        if match is None:
            buffer.append(line)
            continue
        # flush trước khi đổi path: buffer thuộc về heading đang mở.
        flush()
        level = len(match.group(1))
        path[:] = [item for item in path if item[0] < level]
        path.append((level, match.group(2).strip()))

    flush()
    return sections


def _breadcrumb(heading_path: list[str]) -> str:
    """Dòng ngữ cảnh gắn đầu chunk, giới hạn tối đa 1/4 ngân sách chunk."""
    limit = min(MAX_BREADCRUMB_CHARS, max(CHUNK_SIZE // 4, 0))
    parts = [_shorten(title, MAX_HEADING_CHARS) for title in heading_path if title.strip()]
    return _shorten(BREADCRUMB_SEPARATOR.join(parts), limit)


def _shorten(text: str, limit: int) -> str:
    if limit <= 1:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _split_body(body: str, breadcrumb: str) -> list[str]:
    """Recursive-split phần thân, chừa chỗ cho breadcrumb trong CHUNK_SIZE."""
    budget = CHUNK_SIZE - (len(breadcrumb) + 2 if breadcrumb else 0)
    if len(body) <= budget:
        return [body]

    overlap = max(0, min(CHUNK_OVERLAP, budget // 3))
    splitter = _get_splitter(budget, overlap)
    return [text for text in (part.strip() for part in splitter.split_text(body)) if text]


def _get_splitter(chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    key = (chunk_size, chunk_overlap)
    if key not in _splitters:
        _splitters[key] = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SPLIT_SEPARATORS,
            keep_separator=True,
        )
    return _splitters[key]


# ---------------------------------------------------------------------------
# Embed + index
# ---------------------------------------------------------------------------
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk, giữ nguyên các field theo contract."""
    if not chunks:
        return []
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"embed_texts trả {len(vectors)} vector cho {len(chunks)} chunk."
        )
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB và dọn chunk mồ côi của cùng tài liệu."""
    if not chunks:
        return

    collection = get_collection()
    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_chroma_metadata(chunk["metadata"]) for chunk in batch],
        )

    _prune_orphans(collection, chunks)


def _chroma_metadata(metadata: dict) -> dict:
    """Chroma chỉ nhận scalar — bỏ giá trị None (Task 5 khôi phục lại key url)."""
    return {key: value for key, value in metadata.items() if value is not None}


def _prune_orphans(collection, chunks: list[dict]) -> int:
    """Xóa chunk cũ của những tài liệu vừa index nhưng không còn trong lần này.

    Khi một tài liệu được chuẩn hóa lại và ngắn đi, các chunk index cao còn sót
    trong collection sẽ thành dữ liệu mồ côi; upsert không tự dọn giúp.
    """
    doc_ids = sorted({chunk["metadata"]["doc_id"] for chunk in chunks})
    if not doc_ids:
        return 0

    where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
    existing = collection.get(where=where, include=[])
    stale = sorted(set(existing.get("ids") or []) - {chunk["id"] for chunk in chunks})
    if stale:
        collection.delete(ids=stale)
    return len(stale)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from {STANDARDIZED_DIR}")

    chunks = chunk_documents(documents)
    lengths = [len(chunk["content"]) for chunk in chunks]
    print(
        f"Chunked into {len(chunks)} chunks "
        f"(method={CHUNKING_METHOD}, size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
        f"avg={sum(lengths) // len(lengths)} chars, max={max(lengths)})"
    )

    print(f"Embedding with {EMBEDDING_PROVIDER}:{EMBEDDING_MODEL} ...")
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)

    collection = get_collection()
    print(f"Indexed {len(embedded_chunks)} chunks")
    print(f"Collection '{COLLECTION_NAME}' now holds {collection.count()} chunks")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_pipeline()
