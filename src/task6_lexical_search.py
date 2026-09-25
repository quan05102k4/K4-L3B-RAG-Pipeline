"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.

CORPUS được nạp lại đúng những chunk Task 4 đã index (đọc từ ChromaDB), nên hai
đường tìm kiếm luôn xếp hạng trên cùng một tập tài liệu. Nếu vector store chưa
được dựng, corpus được tái tạo bằng load_documents() + chunk_documents() — cùng
thuật toán chunk, cùng ID, chỉ bỏ qua bước embedding vì BM25 không cần vector.

Tokenizer giữ nguyên mã văn bản ("12/2022/nđ-cp") thành một token đồng thời phát
sinh thêm các thành phần rời, nhờ vậy BM25 khớp được cả khi người dùng gõ đầy đủ
số hiệu lẫn khi chỉ nhớ một phần.

Chạy:
    python -m src.task6_lexical_search "câu hỏi của bạn"
"""

from __future__ import annotations

import re
import unicodedata

from .task4_chunking_indexing import (
    chunk_documents,
    get_collection,
    load_documents,
)


CORPUS: list[dict] = []

# Từ thường khớp đúng phần \w+; mã văn bản như "12/2022/nđ-cp" khớp trọn một token.
TOKEN_PATTERN = re.compile(r"\w+(?:[/\-]\w+)*")
COMPOUND_SPLIT = re.compile(r"[/\-]")

_index_cache: tuple[list[dict], object, list[set[str]]] | None = None


def tokenize(text: str) -> list[str]:
    """Chuẩn hóa NFC, hạ chữ thường và tách token giữ được mã văn bản."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(unicodedata.normalize("NFC", text).lower()):
        token = match.group(0)
        tokens.append(token)
        if "/" in token or "-" in token:
            tokens.extend(part for part in COMPOUND_SPLIT.split(token) if part)
    return tokens


def load_corpus() -> list[dict]:
    """Nạp đúng corpus chunk của Task 4, ưu tiên đọc từ ChromaDB."""
    corpus = _corpus_from_vectorstore()
    if corpus:
        return corpus
    # Vector store chưa dựng: tái tạo chunk trực tiếp, ID vẫn trùng khớp.
    return chunk_documents(load_documents())


def _corpus_from_vectorstore() -> list[dict]:
    try:
        payload = get_collection().get(include=["documents", "metadatas"])
    except Exception:  # noqa: BLE001 — thiếu/hỏng vector store thì rơi về tái tạo chunk
        return []

    corpus: list[dict] = []
    for item_id, content, metadata in zip(
        payload.get("ids") or [],
        payload.get("documents") or [],
        payload.get("metadatas") or [],
    ):
        if not content:
            continue
        metadata = dict(metadata or {})
        # Chroma không lưu None nên `url` phải được khôi phục về đúng contract.
        metadata.setdefault("url", None)
        corpus.append({"id": item_id, "content": content, "metadata": metadata})
    # Thứ tự ổn định để BM25 phá hòa điểm một cách xác định giữa các lần chạy.
    corpus.sort(key=lambda item: item["id"])
    return corpus


def get_corpus() -> list[dict]:
    """Trả CORPUS, nạp lười ở lần dùng đầu tiên."""
    global CORPUS
    if not CORPUS:
        CORPUS = load_corpus()
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Corpus rỗng — chạy `python -m src.task4_chunking_indexing` trước.")
    return BM25Okapi([tokenize(item["content"]) for item in corpus])


def _get_index(corpus: list[dict]) -> tuple[object, list[set[str]]]:
    """Cache index và token set theo đúng đối tượng corpus đang dùng."""
    global _index_cache
    if _index_cache is None or _index_cache[0] is not corpus:
        token_sets = [set(tokenize(item["content"])) for item in corpus]
        _index_cache = (corpus, build_bm25_index(corpus), token_sets)
    return _index_cache[1], _index_cache[2]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    tokens = tokenize(query) if query else []
    if not tokens or top_k <= 0:
        return []
    corpus = get_corpus()
    if not corpus:
        return []

    index, token_sets = _get_index(corpus)
    scores = index.get_scores(tokens)
    query_tokens = set(tokens)

    # Lọc theo term overlap chứ không theo `score > 0`: BM25Okapi cho IDF bằng 0
    # với từ xuất hiện ở khoảng một nửa corpus, nên một chunk khớp đúng từ khóa
    # vẫn có thể mang điểm 0 và bị loại oan. Ngược lại chunk không chia sẻ token
    # nào với query thì chắc chắn là nhiễu cho bước fusion.
    candidates = [
        position
        for position in range(len(corpus))
        if query_tokens & token_sets[position]
    ]
    # Phá hòa điểm theo ID để thứ hạng ổn định giữa các lần chạy.
    candidates.sort(key=lambda position: (-scores[position], corpus[position]["id"]))

    return [
        {
            "id": corpus[position]["id"],
            "content": corpus[position]["content"],
            "score": float(scores[position]),
            "metadata": corpus[position]["metadata"],
            "retrieval_method": "bm25",
        }
        for position in candidates[:top_k]
    ]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "Nghị định 12/2022/NĐ-CP xử phạt vi phạm về làm thêm giờ"
    print(f"Query: {question}")
    print(f"Corpus: {len(get_corpus())} chunks\n")
    for rank, result in enumerate(lexical_search(question, top_k=5), start=1):
        breadcrumb = result["metadata"].get("section_path") or ""
        body = result["content"].removeprefix(breadcrumb).strip()
        print(f"{rank}. [{result['score']:.4f}] {result['id']}")
        print(f"   {breadcrumb}")
        print(f"   {body[:160]}...\n".replace("\n", " ") + "\n")
