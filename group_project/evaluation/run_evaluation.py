"""
A/B evaluation runner — Config A (dense-only) vs Config B (hybrid + RRF).

Chỉ một biến được đổi giữa hai lần chạy: `use_reranking` của
``src.task9_retrieval_pipeline.retrieve``. Golden dataset, generator, evaluator,
prompt, top_k, score_threshold và nhánh fallback đều dùng chung. Nếu đổi thêm
biến nào nữa thì delta B−A không còn nói được RRF đóng góp gì.

Generation không viết lại: module này import thẳng SYSTEM_PROMPT,
_citation_order, reorder_for_llm, format_context và call_llm của Task 10, nên
prompt đưa vào LLM ở hai arm là cùng một đoạn code chứ không phải bản sao chép
có thể trôi khỏi bản gốc. Phần duy nhất phải viết lại là lời gọi retrieve(), vì
generate_with_citation() không nhận cờ use_reranking — contract test khoá
signature của nó ở ["query", "top_k"].

Ngoài bốn metric RAGAS, mỗi case còn được đo `evidence_hit`: expected_context
của golden case có nằm trong chunk nào đã lấy về không. Đây là tín hiệu xác
định, không qua LLM judge, dùng để tách lỗi retrieval khỏi lỗi generation khi
đọc worst performers — recall thấp mà evidence_hit=False là lỗi retrieval, còn
evidence_hit=True mà faithfulness thấp là lỗi generation hoặc prompt.

Chạy:
    python -m group_project.evaluation.run_evaluation            # cả hai config
    python -m group_project.evaluation.run_evaluation --retrieval-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task4_chunking_indexing import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    embed_texts,
    get_collection,
)
from src.task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve  # noqa: E402
from src.task10_generation import (  # noqa: E402
    CITATION_BRACKETS,
    LLM_MODEL,
    LLM_PROVIDER,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    _citation_order,
    call_llm,
    format_context,
    reorder_for_llm,
)


load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden_dataset.json"
RUNS_PATH = EVAL_DIR / "runs.json"
SCORES_PATH = EVAL_DIR / "scores.json"

TOP_K = 5

# Judge tách khỏi generator, vì hai lý do đã đo được chứ không phải theo lệ:
#
# 1. Self-preference bias — chấm bằng chính model đã sinh câu trả lời làm
#    faithfulness và answer relevance tuyệt đối cao hơn thực tế.
# 2. Ngân sách token — chấm 4 metric tốn ~14.750 token/case trên corpus này,
#    tức ~590k cho 20 case × 2 config. Groq free tier cho 200k/ngày/model, mà
#    40 lời gọi generation đã ăn gần hết hạn mức của model sinh câu trả lời.
#    Dồn cả hai việc vào một model thì run chấm chết giữa chừng vì 429.
#
# Gemini nói được giao thức OpenAI qua base_url dưới đây nên dùng chung client
# `openai` với generator, không phải thêm dependency. Cả ba biến đọc từ .env
# nên đổi judge không cần sửa code.
#
# Default là flash-lite chứ không phải flash, để chạy lại mà không đặt biến môi
# trường vẫn khớp judge đã dùng trong RESULT.md (`scores.json` ghi lại
# `judge_model` của mỗi lần chấm). Free tier của gemini-3.5-flash chỉ cho 20
# request/ngày, trong khi chấm 4 metric cho 20 case x 2 config cần ~360 request.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemini-3.5-flash-lite")
JUDGE_BASE_URL = os.getenv(
    "EVAL_JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
JUDGE_KEY_NAME = os.getenv("EVAL_JUDGE_KEY_NAME", "GEMINI_API_KEY")
# Provider free tier nào cũng chặn theo token/phút; 4 worker là mức chạy hết 80
# job mà không sinh 429. Tăng lên thì nhanh hơn nhưng NaN nhiều hơn, và một
# metric NaN làm hỏng phép so A/B chứ không chỉ làm chậm.
MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS") or 4)

CONFIGS = {
    "A": {"label": "dense-only", "use_reranking": False},
    "B": {"label": "hybrid + RRF", "use_reranking": True},
}

METRIC_KEYS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevance",
    "context_recall": "context_recall",
    "llm_context_precision_with_reference": "context_precision",
}


# --------------------------------------------------------------------------
# Run metadata — báo cáo phải tái lập được, nên ghi lại cả code lẫn corpus.
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — không có git thì báo cáo vẫn chạy được
        return "unknown"


def corpus_fingerprint() -> dict:
    """Hash nội dung corpus đã chuẩn hoá.

    Thư mục data/ không được commit, nên commit hash của code không xác định
    được corpus. Hash này mới là thứ ràng buộc kết quả với dữ liệu: đổi một ký
    tự trong bất kỳ file .md nào là digest đổi theo.
    """
    files = sorted((ROOT / "data" / "standardized").rglob("*.md"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "files": len(files),
        "sha256": digest.hexdigest()[:16],
        "chunks_indexed": get_collection().count(),
    }


def run_metadata(golden_size: int) -> dict:
    import chromadb
    import ragas

    return {
        "evaluation_date": date.today().isoformat(),
        "code_commit": _git("rev-parse", "--short", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain")),
        "corpus": corpus_fingerprint(),
        "golden_dataset_size": golden_size,
        "framework": f"ragas {ragas.__version__}",
        "chromadb": chromadb.__version__,
        "generator_model": f"{LLM_PROVIDER}:{LLM_MODEL}",
        "generator_params": {"temperature": TEMPERATURE, "top_p": TOP_P},
        "judge_model": JUDGE_MODEL,
        "embedding_model": f"{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}",
        "chunking": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "pageindex_enabled": bool(os.getenv("PAGEINDEX_API_KEY", "").strip()),
    }


# --------------------------------------------------------------------------
# Retrieval + generation — cùng một code path, khác đúng cờ use_reranking.
# --------------------------------------------------------------------------

MARKDOWN_HEADING = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
MARKDOWN_MARKS = re.compile(r"[*_`>]")
WHITESPACE = re.compile(r"\s+")
EVIDENCE_ANCHOR_CHARS = 200


def _normalize(text: str) -> str:
    """Đưa golden context và chunk content về cùng một dạng so sánh được.

    Ba khác biệt phải xoá trước khi so, nếu không phép đo báo trượt cả khi bằng
    chứng nằm ngay ở rank 1:

    - Heading marker: golden context chép từ file .md nên giữ ``#### Điều 106``,
      còn chunker thay heading bằng breadcrumb ``... > Điều 106. ...``.
    - Emphasis marker: bản chuẩn hoá còn sót ``_..._`` và ``**...**`` ở một số
      đoạn trích trong news.
    - Xuống dòng: chunker cắt dòng khác với file gốc.
    """
    text = unicodedata.normalize("NFC", text)
    text = MARKDOWN_HEADING.sub("", text)
    text = MARKDOWN_MARKS.sub("", text)
    return WHITESPACE.sub(" ", text).strip().lower()


def evidence_rank(chunks: list[dict], expected_context: str) -> int | None:
    """Vị trí 1-based của chunk đầu tiên chứa expected_context, None nếu trượt.

    Ưu tiên khớp trọn expected_context. Khi golden context dài hơn phần thân một
    chunk (case multi_fact), lùi về ``EVIDENCE_ANCHOR_CHARS`` ký tự đầu làm mỏ
    neo: câu hỏi cần trả lời là "retrieval có chạm được vào bằng chứng không",
    không phải "có lấy trọn bằng chứng trong một chunk không".

    Phép đo này không gọi LLM nên xác định hoàn toàn — đó là lý do nó đứng cạnh
    context_recall trong báo cáo: khi hai con số lệch nhau, phần lệch nằm ở
    judge chứ không ở retrieval.
    """
    needle = _normalize(expected_context)
    anchor = needle[:EVIDENCE_ANCHOR_CHARS]
    for rank, chunk in enumerate(chunks, start=1):
        haystack = _normalize(chunk["content"])
        if needle in haystack or anchor in haystack:
            return rank
    return None


def run_case(case: dict, use_reranking: bool, *, generate: bool = True) -> dict:
    """Chạy một golden case qua pipeline và trả về mọi thứ evaluation cần."""
    started = time.perf_counter()
    chunks = retrieve(case["question"], top_k=TOP_K, use_reranking=use_reranking)
    retrieval_seconds = time.perf_counter() - started

    ordered = _citation_order(chunks)
    context = format_context(reorder_for_llm(ordered))

    record = {
        "id": case["id"],
        "question": case["question"],
        "question_type": case.get("question_type"),
        "expected_answer": case["expected_answer"],
        "expected_context": case["expected_context"],
        "source_article": case.get("source_article"),
        "retrieved_ids": [chunk["id"] for chunk in ordered],
        "retrieved_sections": [
            (chunk["metadata"] or {}).get("section", "") for chunk in ordered
        ],
        "retrieved_contexts": [chunk["content"] for chunk in ordered],
        "retrieval_method": ordered[0]["retrieval_method"] if ordered else "none",
        "scores": [round(float(chunk["score"]), 6) for chunk in ordered],
        "evidence_rank": evidence_rank(ordered, case["expected_context"]),
        "context_chars": len(context),
        "retrieval_seconds": round(retrieval_seconds, 4),
    }
    record["evidence_hit"] = record["evidence_rank"] is not None

    if not generate:
        record["answer"] = ""
        record["generation_seconds"] = 0.0
        return record

    started = time.perf_counter()
    try:
        # Cùng SYSTEM_PROMPT, cùng call_llm, cùng cách dựng user message như
        # generate_with_citation() — chỉ retrieve() ở trên là khác giữa hai arm.
        answer = call_llm(
            SYSTEM_PROMPT, f"Context:\n{context}\n\nCâu hỏi: {case['question']}"
        )
    except Exception as error:  # noqa: BLE001 — một case lỗi không được giết cả run
        answer = ""
        record["generation_error"] = f"{type(error).__name__}: {error}"
    record["generation_seconds"] = round(time.perf_counter() - started, 4)
    # Cùng bước chuẩn hoá ngoặc CJK mà generate_with_citation() áp dụng trước
    # khi trả về UI. Không có nó thì câu trả lời được chấm khác câu trả lời
    # người dùng thấy, dù chỉ ở nhãn citation.
    record["answer"] = answer.strip().translate(CITATION_BRACKETS)
    return record


def run_config(golden: list[dict], config_id: str, *, generate: bool = True) -> list[dict]:
    config = CONFIGS[config_id]
    records = []
    for index, case in enumerate(golden, start=1):
        record = run_case(case, config["use_reranking"], generate=generate)
        records.append(record)
        hit = f"rank {record['evidence_rank']}" if record["evidence_hit"] else "MISS"
        print(
            f"  [{config_id}] {index:>2}/{len(golden)} {case['id']} "
            f"evidence={hit:<7} retrieval={record['retrieval_seconds']:.2f}s "
            f"gen={record['generation_seconds']:.2f}s",
            flush=True,
        )
    return records


# --------------------------------------------------------------------------
# RAGAS scoring
# --------------------------------------------------------------------------

def build_judge():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    key = os.getenv(JUDGE_KEY_NAME, "").strip()
    if not key:
        raise ValueError(f"Thiếu {JUDGE_KEY_NAME} trong .env cho evaluator.")
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_MODEL,
            api_key=key,
            base_url=JUDGE_BASE_URL,
            # Judge phải xác định nhất có thể: cùng input thì cùng điểm ở hai arm.
            temperature=0.0,
            timeout=180,
            max_retries=3,
        )
    )


def build_judge_embeddings():
    """Answer relevancy cần embedding; dùng lại đúng model đã index corpus."""
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    class PipelineEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return embed_texts(list(texts))

        def embed_query(self, text):
            return embed_texts([text])[0]

    return LangchainEmbeddingsWrapper(PipelineEmbeddings())


def score_records(records: list[dict]) -> list[dict]:
    """Chấm 4 metric RAGAS và gắn điểm vào từng record."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    samples = [
        SingleTurnSample(
            user_input=record["question"],
            response=record["answer"] or " ",
            retrieved_contexts=record["retrieved_contexts"] or [" "],
            reference=record["expected_answer"],
            reference_contexts=[record["expected_context"]],
        )
        for record in records
    ]

    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            # strictness=1: mặc định của RAGAS là 3, tức xin LLM sinh 3 câu hỏi
            # ngược bằng tham số `n=3`. Groq từ chối n>1 (400 "'n' : number must
            # be at most 1"), nên để mặc định thì metric này hỏng chứ không phải
            # chặt hơn. Đổi lại: answer relevance mỗi case dựa trên một câu hỏi
            # ngược, nhiễu hơn — nhưng cả hai arm chịu cùng mức nhiễu đó.
            ResponseRelevancy(strictness=1),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=build_judge(),
        embeddings=build_judge_embeddings(),
        run_config=RunConfig(timeout=300, max_retries=5, max_workers=MAX_WORKERS),
        raise_exceptions=False,
        show_progress=True,
    )

    frame = result.to_pandas()
    for record, (_, row) in zip(records, frame.iterrows()):
        record["metrics"] = {
            # NaN != NaN là cách nhận ra case judge không chấm được; giữ None để
            # aggregate() loại nó ra thay vì kéo trung bình xuống 0.
            name: (None if row[key] != row[key] else round(float(row[key]), 4))
            for key, name in METRIC_KEYS.items()
        }
    return records


def _metric_instances(judge, embeddings) -> dict:
    """Một instance metric cho mỗi tên, đã gắn judge — dùng cho lượt chấm bù."""
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    metrics = {
        "faithfulness": Faithfulness(),
        "answer_relevance": ResponseRelevancy(strictness=1),
        "context_recall": LLMContextRecall(),
        "context_precision": LLMContextPrecisionWithReference(),
    }
    for metric in metrics.values():
        metric.llm = judge
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
    return metrics


def fill_missing_scores(runs: dict, attempts: int = 3) -> int:
    """Chấm lại từng ô None, trả về số ô vá được.

    Judge nhẹ thỉnh thoảng trả JSON hỏng (OUTPUT_PARSING_FAILURE), và số ô hỏng
    không chia đều giữa hai arm — lần chạy này rơi 2 ô ở A nhưng 17 ô ở B. Bỏ
    qua thì n của hai arm lệch nhau, mà chấm lại toàn bộ thì tốn đúng số token
    đã tiêu. Ở đây chỉ gọi lại đúng metric của đúng case còn thiếu.
    """
    import asyncio

    from ragas import SingleTurnSample

    judge = build_judge()
    metrics = _metric_instances(judge, build_judge_embeddings())

    filled = 0
    for config_id, records in runs.items():
        for record in records:
            gaps = [name for name, value in record["metrics"].items() if value is None]
            for name in gaps:
                sample = SingleTurnSample(
                    user_input=record["question"],
                    response=record["answer"] or " ",
                    retrieved_contexts=record["retrieved_contexts"] or [" "],
                    reference=record["expected_answer"],
                    reference_contexts=[record["expected_context"]],
                )
                for attempt in range(1, attempts + 1):
                    try:
                        score = asyncio.run(
                            metrics[name].single_turn_ascore(sample)
                        )
                    except Exception as error:  # noqa: BLE001 — vá được ô nào hay ô đó
                        print(f"  [{config_id}] {record['id']}.{name} thử {attempt}: "
                              f"{type(error).__name__}", flush=True)
                        continue
                    if score == score:  # loại NaN
                        record["metrics"][name] = round(float(score), 4)
                        filled += 1
                        print(f"  [{config_id}] {record['id']}.{name} = "
                              f"{record['metrics'][name]}", flush=True)
                        break
    return filled


def aggregate(records: list[dict]) -> dict:
    """Trung bình từng metric, bỏ qua case judge trả NaN và đếm chúng."""
    summary = {}
    for name in METRIC_KEYS.values():
        values = [
            record["metrics"][name]
            for record in records
            if record.get("metrics", {}).get(name) is not None
        ]
        summary[name] = round(sum(values) / len(values), 4) if values else None
        summary[f"{name}_n"] = len(values)
    scored = [summary[name] for name in METRIC_KEYS.values() if summary[name] is not None]
    summary["average"] = round(sum(scored) / len(scored), 4) if scored else None
    summary["evidence_hit_rate"] = round(
        sum(record["evidence_hit"] for record in records) / len(records), 4
    )
    # Chỉ tính trên case có hit: rank của case trượt là vô cực, thay bằng một
    # con số bịa (top_k + 1) sẽ làm trung bình phụ thuộc vào top_k chứ không
    # phải vào chất lượng xếp hạng. Hit rate ở trên đã đếm phần trượt rồi.
    ranks = [record["evidence_rank"] for record in records if record["evidence_hit"]]
    summary["mean_evidence_rank"] = round(sum(ranks) / len(ranks), 3) if ranks else None
    summary["evidence_at_1"] = sum(rank == 1 for rank in ranks)
    summary["mean_retrieval_seconds"] = round(
        sum(record["retrieval_seconds"] for record in records) / len(records), 4
    )
    summary["mean_generation_seconds"] = round(
        sum(record["generation_seconds"] for record in records) / len(records), 4
    )
    summary["mean_context_chars"] = round(
        sum(record["context_chars"] for record in records) / len(records), 1
    )
    return summary


def paired_delta(runs: dict) -> dict:
    """Delta B−A tính trên đúng những case cả hai arm đều chấm được.

    Judge thỉnh thoảng trả NaN vì rate limit hoặc timeout, và không nhất thiết
    trượt cùng case ở hai arm. Lấy hiệu của hai trung bình toàn phần khi đó là
    so hai tập case khác nhau: một phần delta đến từ việc case nào bị rớt chứ
    không từ retrieval strategy. Hàm này chỉ giữ case có điểm ở cả A lẫn B, nên
    delta đọc được đúng như một phép so ghép cặp.
    """
    index = {config_id: {r["id"]: r for r in records} for config_id, records in runs.items()}
    shared = sorted(set(index["A"]) & set(index["B"]))

    deltas = {}
    for name in METRIC_KEYS.values():
        pairs = [
            (index["A"][case_id]["metrics"][name], index["B"][case_id]["metrics"][name])
            for case_id in shared
            if index["A"][case_id].get("metrics", {}).get(name) is not None
            and index["B"][case_id].get("metrics", {}).get(name) is not None
        ]
        if not pairs:
            deltas[name] = {"n": 0, "a": None, "b": None, "delta": None}
            continue
        mean_a = sum(a for a, _ in pairs) / len(pairs)
        mean_b = sum(b for _, b in pairs) / len(pairs)
        deltas[name] = {
            "n": len(pairs),
            "a": round(mean_a, 4),
            "b": round(mean_b, 4),
            "delta": round(mean_b - mean_a, 4),
            # Đếm case đổi chiều: delta trung bình gần 0 có thể là "không đổi gì"
            # hoặc "đổi nhiều nhưng bù trừ nhau", hai kết luận rất khác nhau.
            "improved": sum(b > a for a, b in pairs),
            "regressed": sum(b < a for a, b in pairs),
        }
    return {"shared_cases": len(shared), "metrics": deltas}


def print_summary(summaries: dict) -> None:
    a, b = summaries["A"], summaries["B"]
    print("\n=== Overall scores ===")
    print(f"{'metric':<22}{'A':>10}{'B':>10}{'B-A':>10}")
    for name in list(METRIC_KEYS.values()) + [
        "average", "evidence_hit_rate", "mean_evidence_rank", "evidence_at_1"
    ]:
        va, vb = a.get(name), b.get(name)
        delta = f"{vb - va:+.4f}" if va is not None and vb is not None else "n/a"
        print(f"{name:<22}{_fmt(va):>10}{_fmt(vb):>10}{delta:>10}")
    for label, key in (
        ("retrieval_s (mean)", "mean_retrieval_seconds"),
        ("generation_s (mean)", "mean_generation_seconds"),
        ("context_chars (mean)", "mean_context_chars"),
    ):
        print(f"{label:<22}{a[key]:>10.3f}{b[key]:>10.3f}{b[key] - a[key]:>+10.3f}")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="A/B evaluation runner")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Chỉ chạy retrieval + evidence hit, không gọi LLM.")
    parser.add_argument("--no-score", action="store_true",
                        help="Chạy pipeline và lưu runs.json, bỏ qua RAGAS.")
    parser.add_argument("--score-only", action="store_true",
                        help="Chấm lại runs.json đã lưu, không chạy lại pipeline.")
    parser.add_argument("--fill-gaps", action="store_true",
                        help="Chấm bù các ô NaN trong scores.json, giữ nguyên ô đã có.")
    args = parser.parse_args()

    if args.fill_gaps:
        saved = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        runs = saved["per_case"]
        print(f"Chấm bù bằng {JUDGE_MODEL}...")
        filled = fill_missing_scores(runs)
        print(f"\nVá được {filled} ô.")
        saved["metadata"]["gap_fill_judge"] = JUDGE_MODEL
        saved["summary"] = {cid: aggregate(recs) for cid, recs in runs.items()}
        saved["paired_delta"] = paired_delta(runs)
        SCORES_PATH.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {SCORES_PATH.relative_to(ROOT)}")
        print_summary(saved["summary"])
        _print_paired(saved["paired_delta"])
        return

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    # Tách chấm khỏi chạy: judge hay hỏng vì rate limit hơn hẳn pipeline, mà
    # chạy lại generation thì câu trả lời đổi (temperature 0.3) nên số cũ và số
    # mới không còn so được với nhau. --score-only chấm lại đúng những câu trả
    # lời đã sinh.
    if args.score_only:
        saved = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
        metadata, runs = saved["metadata"], saved["runs"]
        metadata["judge_model"] = JUDGE_MODEL
        metadata["scored_date"] = date.today().isoformat()
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return _score_and_save(metadata, runs)

    generate = not args.retrieval_only
    metadata = run_metadata(len(golden))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    # Nạp sẵn BM25 corpus và embedding model để latency đo được không gánh chi
    # phí khởi động một lần của lời gọi đầu tiên.
    print("\nWarming up retrieval...")
    retrieve("khởi động", top_k=TOP_K, use_reranking=True)

    runs = {}
    for config_id in ("A", "B"):
        print(f"\n--- Config {config_id}: {CONFIGS[config_id]['label']} ---")
        runs[config_id] = run_config(golden, config_id, generate=generate)

    RUNS_PATH.write_text(
        json.dumps({"metadata": metadata, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {RUNS_PATH.relative_to(ROOT)}")

    if args.retrieval_only or args.no_score:
        for config_id, records in runs.items():
            hits = sum(record["evidence_hit"] for record in records)
            print(f"Config {config_id}: evidence hit {hits}/{len(records)}")
        return

    _score_and_save(metadata, runs)


def _score_and_save(metadata: dict, runs: dict) -> None:
    summaries = {}
    for config_id in ("A", "B"):
        print(f"\n--- Scoring config {config_id} (judge: {JUDGE_MODEL}) ---")
        score_records(runs[config_id])
        summaries[config_id] = aggregate(runs[config_id])

    paired = paired_delta(runs)
    SCORES_PATH.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary": summaries,
                "paired_delta": paired,
                "per_case": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {SCORES_PATH.relative_to(ROOT)}")
    print_summary(summaries)
    _print_paired(paired)


def _print_paired(paired: dict) -> None:
    print("\n=== Paired delta (chỉ case chấm được ở cả hai arm) ===")
    print(f"{'metric':<22}{'n':>4}{'A':>10}{'B':>10}{'B-A':>10}{'up/down':>10}")
    for name, row in paired["metrics"].items():
        if row["delta"] is None:
            print(f"{name:<22}{0:>4}{'n/a':>10}{'n/a':>10}{'n/a':>10}{'-':>10}")
            continue
        churn = f"{row['improved']}/{row['regressed']}"
        print(
            f"{name:<22}{row['n']:>4}{row['a']:>10.4f}{row['b']:>10.4f}"
            f"{row['delta']:>+10.4f}{churn:>10}"
        )


if __name__ == "__main__":
    main()
