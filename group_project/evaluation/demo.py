"""
Demo script — ba phần bắt buộc khi nộp bài, chạy bằng một lệnh.

    1. Query đúng domain   -> dense score trên ngưỡng, trả lời có citation.
    2. Query ngoài domain  -> dense score dưới ngưỡng, thử fallback, safe refusal.
    3. A/B                 -> Config A (dense-only) vs Config B (hybrid + RRF)
                              trên chính query demo, kèm tổng hợp 20 case đã
                              chấm trong scores.json.

Script không đo lại metric và không sinh dữ liệu mới cho báo cáo: phần A/B trên
toàn golden dataset thuộc về `run_evaluation.py`. Ở đây chỉ dựng lại đúng những
gì người xem demo cần thấy tận mắt — đường đi của một query, quyết định fallback
đọc từ cosine score gốc, và citation trỏ về đúng chunk trong `sources`.

Chạy:
    python -m group_project.evaluation.demo              # cả ba phần
    python -m group_project.evaluation.demo --no-llm     # bỏ phần gọi provider
    python -m group_project.evaluation.demo --part ab    # chỉ một phần
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task4_chunking_indexing import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve  # noqa: E402
from src.task10_generation import (  # noqa: E402
    LLM_MODEL,
    LLM_PROVIDER,
    REFUSAL_ANSWER,
    generate_with_citation,
)


SCORES_PATH = Path(__file__).resolve().parent / "scores.json"

# Nhãn citation trong answer. Không có nhãn nào nghĩa là câu trả lời không neo
# vào chunk nào trong context — với query ngoài domain đó chính là hành vi đúng.
CITATION_PATTERN = re.compile(r"\[Document\s+(\d+)\]", re.IGNORECASE)

TOP_K = 5
CANDIDATE_K = TOP_K * 2

IN_DOMAIN_QUERY = "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?"
OUT_DOMAIN_QUERY = "Công thức nấu phở bò Hà Nội cần những nguyên liệu gì?"


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_routing(query: str) -> float:
    """In quyết định fallback: so cosine score gốc của dense với threshold."""
    dense = semantic_search(query, top_k=CANDIDATE_K)
    best = max((item["score"] for item in dense), default=0.0)
    above = best >= SCORE_THRESHOLD

    print(f"Query: {query}")
    print(f"  best dense cosine = {best:.4f}   SCORE_THRESHOLD = {SCORE_THRESHOLD}")
    route = "giữ hybrid, không gọi fallback" if above else "thử PageIndex fallback"
    print(f"  -> {'TRÊN' if above else 'DƯỚI'} ngưỡng => {route}")
    if not above:
        print("     PAGEINDEX_API_KEY trống => fallback trả rỗng => giữ hybrid result")
    return best


def classify_refusal(result: dict) -> str:
    """Phân loại cách hệ thống từ chối.

    Hai đường từ chối khác nhau và phải đọc khác nhau:

    - "deterministic": pipeline tự trả REFUSAL_ANSWER vì không có chunk nào
      hoặc provider lỗi. Không phụ thuộc model.
    - "model": model đọc context thấy không trả lời được nên từ chối bằng lời
      của nó. Dấu hiệu kiểm được bằng máy là answer không trích nhãn
      [Document N] nào — tức không khẳng định điều gì dựa trên context.

    Query ngoài domain trên corpus này đi đường thứ hai: retrieve() vẫn trả về
    5 chunk (hybrid vẫn xếp hạng được dù chunk nào cũng không liên quan), nên
    nhánh deterministic không kích hoạt và SYSTEM_PROMPT là thứ chặn.
    """
    if result["answer"].startswith(REFUSAL_ANSWER):
        return "deterministic"
    if result["sources"] and not CITATION_PATTERN.search(result["answer"]):
        return "model"
    return "no"


def show_answer(query: str) -> dict:
    """Gọi pipeline đầy đủ và in answer kèm nguồn đã dùng."""
    result = generate_with_citation(query, TOP_K)
    refusal = classify_refusal(result)
    refusal_label = {
        "deterministic": "CO (safe refusal cua pipeline)",
        "model": "CO (model tu choi, khong trich nhan nao)",
        "no": "khong",
    }[refusal]

    print(
        f"\nretrieval_source = {result['retrieval_source']}"
        f" | sources = {len(result['sources'])}"
        f" | tu choi = {refusal_label}"
    )
    print("\n--- Answer ---")
    print(result["answer"])

    if result["sources"]:
        print("\n--- Nguồn đã đưa vào context (nhãn khớp sources[N-1]) ---")
        cited = {int(label) for label in CITATION_PATTERN.findall(result["answer"])}
        for index, source in enumerate(result["sources"], start=1):
            metadata = source["metadata"]
            mark = "duoc trich" if index in cited else "khong duoc trich"
            print(
                f"[Document {index}] {mark}"
                f" | {source['retrieval_method']} {source['score']:.6f}"
            )
            print(f"    {metadata['title']}")
            print(f"    file={metadata['source']}  id={source['id']}")
            if metadata.get("url"):
                print(f"    url={metadata['url']}")

        phantom = cited - set(range(1, len(result["sources"]) + 1))
        if phantom:
            print(
                "[!] Answer trích nhãn không có trong sources: "
                + ", ".join(f"[Document {index}]" for index in sorted(phantom))
            )
    return result


def demo_in_domain(use_llm: bool) -> None:
    rule("1. QUERY DUNG DOMAIN")
    show_routing(IN_DOMAIN_QUERY)
    if use_llm:
        show_answer(IN_DOMAIN_QUERY)
    else:
        print("\n(--no-llm: bỏ qua bước generation)")


def demo_out_of_domain(use_llm: bool) -> None:
    rule("2. QUERY NGOAI DOMAIN")
    show_routing(OUT_DOMAIN_QUERY)
    if use_llm:
        result = show_answer(OUT_DOMAIN_QUERY)
        if classify_refusal(result) == "no":
            print(
                "\n[!] Query ngoài domain nhưng answer vẫn trích dẫn nguồn — "
                "xem lại SYSTEM_PROMPT hoặc ngưỡng."
            )
    else:
        print("\n(--no-llm: bỏ qua bước generation)")


def compare_ab(query: str) -> None:
    """Cùng query, cùng top_k, chỉ đổi cờ use_reranking."""
    config_a = retrieve(query, top_k=TOP_K, use_reranking=False)
    config_b = retrieve(query, top_k=TOP_K, use_reranking=True)

    ids_a = [item["id"] for item in config_a]
    ids_b = [item["id"] for item in config_b]
    rank_a = {item_id: rank for rank, item_id in enumerate(ids_a, start=1)}

    print(f"\nQuery: {query}")
    print(f"  {len(set(ids_a) & set(ids_b))}/{TOP_K} chunk trùng nhau giữa hai config")
    print("\n  Config A — dense-only (use_reranking=False)")
    for rank, item in enumerate(config_a, start=1):
        print(f"    A{rank}. [{item['retrieval_method']} {item['score']:.6f}] {item['id']}")

    print("\n  Config B — hybrid + RRF (use_reranking=True)")
    for rank, item in enumerate(config_b, start=1):
        moved = rank_a.get(item["id"])
        origin = f"A#{moved}" if moved else "moi (BM25 keo len)"
        print(
            f"    B{rank}. [{item['retrieval_method']} {item['score']:.6f}] "
            f"{item['id']}  <- {origin}"
        )

    dropped = [item_id for item_id in ids_a if item_id not in ids_b]
    if dropped:
        print(f"\n  Bị B loại khỏi top-{TOP_K}: {', '.join(dropped)}")


def show_scored_ab() -> None:
    """Tổng hợp A/B đã chấm trên toàn golden dataset (scores.json)."""
    if not SCORES_PATH.exists():
        print("\n(scores.json chưa có — chạy run_evaluation trước)")
        return

    scores = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
    metadata = scores["metadata"]
    summary = scores["summary"]

    print(
        f"\nTổng hợp {metadata['golden_dataset_size']} case "
        f"(judge: {metadata['judge_model']}, generator: {metadata['generator_model']})"
    )
    print(
        f"  corpus sha256={metadata['corpus']['sha256']} "
        f"chunks={metadata['corpus']['chunks_indexed']} top_k={metadata['top_k']}"
    )

    print(f"\n  {'Metric':<22}{'Config A':>10}{'Config B':>10}{'Delta B-A':>12}")
    print("  " + "-" * 54)
    rows = [
        ("Faithfulness", "faithfulness"),
        ("Answer relevance", "answer_relevance"),
        ("Context recall", "context_recall"),
        ("Context precision", "context_precision"),
        ("Average", "average"),
        ("Evidence hit rate", "evidence_hit_rate"),
        ("Evidence rank TB", "mean_evidence_rank"),
        ("Retrieval latency (s)", "mean_retrieval_seconds"),
    ]
    for label, key in rows:
        value_a, value_b = summary["A"][key], summary["B"][key]
        print(f"  {label:<22}{value_a:>10.4f}{value_b:>10.4f}{value_b - value_a:>+12.4f}")

    print(
        "\n  Phân tích đầy đủ (worst performers, root cause, recommendations):"
        "\n  group_project/evaluation/RESULT.md"
    )


def demo_ab() -> None:
    rule("3. A/B — CONFIG A (dense-only) vs CONFIG B (hybrid + RRF)")
    compare_ab(IN_DOMAIN_QUERY)
    show_scored_ab()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Demo RAG pipeline")
    parser.add_argument(
        "--part",
        choices=["in", "out", "ab", "all"],
        default="all",
        help="Chạy một phần thay vì cả ba.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Chỉ chạy retrieval, không gọi LLM provider (không tốn quota).",
    )
    args = parser.parse_args()
    use_llm = not args.no_llm

    rule("CAU HINH")
    print(f"  Generator : {LLM_PROVIDER} / {LLM_MODEL or 'default cua provider'}")
    print(f"  Embedding : {EMBEDDING_PROVIDER} / {EMBEDDING_MODEL}")
    print(f"  Chunking  : size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")
    print(f"  Retrieval : top_k={TOP_K} score_threshold={SCORE_THRESHOLD}")

    if args.part in ("in", "all"):
        demo_in_domain(use_llm)
    if args.part in ("out", "all"):
        demo_out_of_domain(use_llm)
    if args.part in ("ab", "all"):
        demo_ab()

    print()


if __name__ == "__main__":
    main()
