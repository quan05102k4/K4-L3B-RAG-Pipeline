"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Hai thang đo không so sánh được với nhau: cosine similarity nằm trong [0, 1]
còn BM25 score là số dương không chặn trên, phụ thuộc IDF của corpus. RRF bỏ
hẳn độ lớn score và chỉ đọc vị trí, nên một chunk đứng cao ở cả hai danh sách
thường vượt chunk chỉ mạnh một phía.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.
Task 9 so threshold với cosine score gốc của dense search (xem module đó).

Reranker mô hình hóa (Jina, self-hosted cross-encoder) là hướng mở rộng bonus:
nếu dùng thì phải so sánh với baseline RRF này, không thay thế nó.

Chạy:
    python -m src.task7_reranking "câu hỏi của bạn"
"""

from __future__ import annotations


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult.

    Args:
        ranked_lists: Các danh sách SearchResult đã sort giảm dần theo score
            riêng của từng phương pháp. Vị trí trong list chính là rank.
        top_k: Số kết quả tối đa trả về.
        k: Hằng số làm mượt. k lớn kéo các rank đầu lại gần nhau, giảm ảnh
            hưởng của việc một danh sách xếp hạng sai ở top.

    Returns:
        SearchResult đã gộp, ``retrieval_method="hybrid"``, ID duy nhất, sort
        giảm dần theo RRF score và không vượt ``top_k``.
    """
    if top_k <= 0:
        return []
    if k <= 0:
        raise ValueError("k phải dương để tránh chia cho 0 ở rank đầu tiên")

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        # Một ID chỉ được tính điểm một lần cho mỗi danh sách. ID lặp trong
        # cùng danh sách là lỗi invariant của module sinh ra nó (Task 5/6 đều
        # đã khử trùng); cộng dồn ở đây sẽ thổi phồng điểm và hợp thức hóa dữ
        # liệu hỏng, nên chỉ giữ rank tốt nhất và bỏ qua các lần xuất hiện sau.
        counted: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            if item_id in counted:
                continue
            counted.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            # Lần gặp đầu tiên thắng: danh sách đứng trước trong ranked_lists
            # (dense) giữ vai trò nguồn nội dung chuẩn cho chunk trùng nhau.
            items.setdefault(item_id, item)

    # Phá hòa điểm theo ID để thứ hạng ổn định giữa các lần chạy.
    ranked_ids = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))

    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        # Copy trước khi ghi đè: hai ranked list đầu vào phải giữ nguyên cosine
        # score gốc cho bước quyết định fallback của Task 9, và test phải phân
        # biệt được "kết quả mới" với "mutation tại chỗ".
        result = dict(items[item_id])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    import sys

    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?"
    print(f"Query: {question}\n")

    dense = semantic_search(question, top_k=10)
    sparse = lexical_search(question, top_k=10)
    fused = rerank_rrf([dense, sparse], top_k=5)

    dense_rank = {item["id"]: rank for rank, item in enumerate(dense, start=1)}
    sparse_rank = {item["id"]: rank for rank, item in enumerate(sparse, start=1)}

    print(f"dense: {len(dense)} kết quả | bm25: {len(sparse)} kết quả\n")
    for rank, result in enumerate(fused, start=1):
        positions = (
            f"dense#{dense_rank.get(result['id'], '-')} "
            f"bm25#{sparse_rank.get(result['id'], '-')}"
        )
        print(f"{rank}. [RRF {result['score']:.6f}] ({positions}) {result['id']}")
        print(f"   {result['metadata'].get('section_path', '')}\n")
