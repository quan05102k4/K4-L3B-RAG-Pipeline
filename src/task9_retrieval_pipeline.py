"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Hai quyết định dùng hai score khác nhau, và đó là điểm dễ sai nhất của module:

    - RRF chỉ hợp nhất *thứ hạng*. RRF score nằm quanh 1/61 + 1/62 ≈ 0.033 dù
      chunk khớp hoàn hảo hay chỉ khớp lơ mơ, vì nó chỉ đọc vị trí.
    - Fallback hỏi "dense có tự tin không", nên phải đọc *cosine score gốc*.
      So threshold với RRF score sẽ bật fallback cho mọi query.

SCORE_THRESHOLD đọc từ .env và phải được hiệu chỉnh lại cho từng corpus bằng
một query trong domain và một query ngoài domain (ghi lại trong
group_project/evaluation/RESULT.md). Không có con số nào đúng cho mọi corpus.

Chạy:
    python -m src.task9_retrieval_pipeline "câu hỏi của bạn"
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

# 0.60 là điểm giữa khoảng trống đo được trên corpus lao động với bge-m3:
# query trong domain cho best cosine 0.73–0.80, query ngoài domain 0.35–0.48.
# Số liệu và hai query hiệu chỉnh nằm trong group_project/evaluation/RESULT.md.
# .env ghi đè được để đổi ngưỡng mà không phải sửa code.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.60)
DEFAULT_TOP_K = 5
# Lấy rộng hơn top_k ở hai nhánh để RRF có đủ ứng viên chồng lấn; chunk chỉ
# xuất hiện ở một danh sách vẫn có cơ hội vào top sau khi fuse.
CANDIDATE_MULTIPLIER = 2


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    if not query or not query.strip() or top_k <= 0:
        return []

    candidate_k = top_k * CANDIDATE_MULTIPLIER
    dense = semantic_search(query, top_k=candidate_k)
    sparse = lexical_search(query, top_k=candidate_k)

    # RRF chạy đúng một lần cho cả pipeline.
    hybrid = rerank_rrf([dense, sparse], top_k=top_k) if use_reranking else dense[:top_k]

    # Cosine score gốc, không phải RRF score. dense đã sort giảm dần nên phần tử
    # đầu là tốt nhất; dùng max() để quyết định không phụ thuộc thứ tự đầu vào.
    best_dense_score = max((item["score"] for item in dense), default=0.0)

    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
        except Exception as error:  # noqa: BLE001 — provider ngoài, UI không được chết
            print(
                f"[retrieve] fallback lỗi ({type(error).__name__}: {error}); "
                "giữ hybrid result.",
                file=sys.stderr,
            )
        else:
            # Fallback rỗng nghĩa là PageIndex không tắt nhưng cũng không giúp
            # được gì; hybrid vẫn tốt hơn là trả về tay trắng.
            if fallback:
                return fallback[:top_k]

    return hybrid[:top_k]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "Thời giờ làm việc bình thường tối đa bao nhiêu giờ một ngày?"

    dense_preview = semantic_search(question, top_k=DEFAULT_TOP_K * CANDIDATE_MULTIPLIER)
    best_dense = max((item["score"] for item in dense_preview), default=0.0)

    print(f"Query: {question}")
    print(f"best dense cosine = {best_dense:.4f} | threshold = {SCORE_THRESHOLD}")
    print(f"-> {'trên' if best_dense >= SCORE_THRESHOLD else 'dưới'} ngưỡng\n")

    for rank, result in enumerate(retrieve(question, top_k=DEFAULT_TOP_K), start=1):
        breadcrumb = result["metadata"].get("section_path") or ""
        body = result["content"].removeprefix(breadcrumb).strip()
        print(f"{rank}. [{result['retrieval_method']} {result['score']:.6f}] {result['id']}")
        print(f"   {breadcrumb}")
        print(f"   {body[:160]}...\n".replace("\n", " ") + "\n")
