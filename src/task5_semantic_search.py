"""
Task 5 — Semantic search (dense retrieval).

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.

Dùng chung embed_texts() với Task 4 là ràng buộc bắt buộc: cùng model, cùng
dimension, cùng cách normalize — lệch một trong ba thì khoảng cách cosine vô nghĩa.

Chạy:
    python -m src.task5_semantic_search "câu hỏi của bạn"
"""

from __future__ import annotations

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query or not query.strip() or top_k <= 0:
        return []

    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    seen: set[str] = set()
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        if item_id in seen:
            continue
        seen.add(item_id)
        results.append(
            {
                "id": item_id,
                "content": content,
                "score": _to_similarity(distance),
                "metadata": _restore_metadata(metadata),
                "retrieval_method": "dense",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _to_similarity(distance: float) -> float:
    """Chroma trả cosine distance; Task 9 so threshold với similarity trong [0, 1]."""
    return min(1.0, max(0.0, 1.0 - float(distance)))


def _restore_metadata(metadata: dict | None) -> dict:
    """Khôi phục key `url` — Chroma không lưu được giá trị None nên Task 4 đã lược bỏ."""
    restored = dict(metadata or {})
    restored.setdefault("url", None)
    return restored


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "Thời giờ làm việc bình thường tối đa bao nhiêu giờ một ngày?"
    print(f"Query: {question}\n")
    for rank, result in enumerate(semantic_search(question, top_k=5), start=1):
        breadcrumb = result["metadata"].get("section_path") or ""
        body = result["content"].removeprefix(breadcrumb).strip()
        print(f"{rank}. [{result['score']:.4f}] {result['id']}")
        print(f"   {breadcrumb}")
        print(f"   {body[:160]}...\n".replace("\n", " ") + "\n")
