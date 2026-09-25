"""
Task 8 — PageIndex vectorless fallback.

PageIndex không dùng vector: nó dựng cây mục lục cho từng tài liệu rồi để LLM
đi trên cây đó. Vì vậy nó bù đúng chỗ dense + BM25 yếu — câu hỏi cần đọc ngữ
cảnh dài hoặc suy luận qua nhiều mục thay vì khớp từ khóa.

Luồng:
    1. Đọc PAGEINDEX_API_KEY từ .env (thiếu key -> coi như không bật fallback).
    2. Render mỗi Markdown chuẩn hóa thành PDF rồi upload (API chỉ nhận PDF).
    3. Cache doc_id theo sha256 nội dung để không upload lại và không trả phí lại.
    4. Submit query cho mọi doc_id, poll trong một ngân sách thời gian cố định,
       parse retrieved nodes thành SearchResult có retrieval_method="pageindex".

Vì sao gọi REST trực tiếp thay vì `PageIndexClient`: SDK 0.2.8 gọi `requests`
không kèm `timeout`, nên một kết nối treo sẽ giữ Streamlit lại vô hạn. Ở đây
mỗi request có timeout riêng và cả vòng retrieval có deadline chung. Endpoint
và kiểu lỗi vẫn mượn của SDK để không lệch khi SDK đổi BASE_URL.

Upload bản Markdown chuẩn hóa (không phải PDF gốc trong data/landing) là chủ ý:
PageIndex đọc đúng văn bản mà dense và BM25 đang index, và Nghị định 145/2020
vốn tách làm hai file PDF gốc vẫn vào PageIndex như một tài liệu liền mạch.

Chạy:
    python -m src.task8_pageindex_vectorless           # upload + cache doc_id
    python -m src.task8_pageindex_vectorless "câu hỏi" # thử một truy vấn
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
# Corpus lấy qua load_documents() của Task 4 để PageIndex đọc đúng bộ tài liệu
# mà dense và BM25 đang index.
CACHE_DIR = ROOT_DIR / "data" / "pageindex"
CACHE_PATH = CACHE_DIR / "documents.json"
PDF_DIR = CACHE_DIR / "pdf"

try:  # SDK là dependency chính thức; mượn endpoint và kiểu lỗi của nó.
    from pageindex import PageIndexAPIError
    from pageindex import PageIndexClient as _SDKClient

    _DEFAULT_BASE_URL = _SDKClient.BASE_URL
except ImportError:  # pragma: no cover — chạy được cả khi chưa cài SDK

    class PageIndexAPIError(RuntimeError):
        """Dịch vụ trả về response không dùng được."""

    _DEFAULT_BASE_URL = "https://api.pageindex.ai"


class PageIndexUnavailable(RuntimeError):
    """Không gọi được PageIndex (mạng, timeout, thiếu cấu hình ở phía dịch vụ)."""


BASE_URL = os.getenv("PAGEINDEX_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")

# (connect, read). Upload cần read dài hơn vì phải đẩy cả file lên.
REQUEST_TIMEOUT = (10.0, 30.0)
UPLOAD_TIMEOUT = (10.0, 180.0)
# Ngân sách cho cả vòng retrieval: fallback chỉ được phép làm chậm UI có giới hạn.
RETRIEVAL_BUDGET_SECONDS = float(os.getenv("PAGEINDEX_BUDGET_SECONDS", "45"))
POLL_INTERVAL_SECONDS = 1.5

DONE_STATUSES = {"completed", "complete", "done", "success", "succeeded", "ready"}
FAILED_STATUSES = {"failed", "error", "cancelled", "canceled", "rejected"}

# Tên field của response có thể đổi giữa các version API. Thay vì đoán cứng một
# tên, parser dò theo các container thường gặp rồi mới duyệt nông toàn payload.
NODE_LIST_KEYS = (
    "retrieval",
    "retrieved_nodes",
    "nodes",
    "sources",
    "results",
    "result",
    "data",
    "response",
)
TEXT_KEYS = (
    "text",
    "content",
    "node_text",
    "relevant_content",
    "contents",
    "markdown",
    "node_summary",
    "summary",
)
SCORE_KEYS = ("score", "relevance_score", "relevance", "confidence")
NODE_ID_KEYS = ("node_id", "id", "nodeId")
TITLE_KEYS = ("title", "node_title", "section", "heading")

# fpdf2 không kèm font Unicode; tiếng Việt cần một TTF có dấu.
FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Token ngắn hơn ngần này chắc chắn lọt khung A4 ở cỡ chữ đang dùng, khỏi phải đo.
CHAR_WRAP_PROBE_LENGTH = 20

_warned: set[str] = set()


# ---------------------------------------------------------------------------
# Hạ tầng
# ---------------------------------------------------------------------------
def is_enabled() -> bool:
    """PageIndex chỉ được coi là đang bật khi có API key."""
    return bool(PAGEINDEX_API_KEY)


def _warn(message: str) -> None:
    """In cảnh báo một lần cho mỗi nội dung — fallback chạy trong vòng lặp chat."""
    if message not in _warned:
        _warned.add(message)
        print(f"[pageindex] {message}", file=sys.stderr)


def _request(method: str, path: str, *, timeout=REQUEST_TIMEOUT, **kwargs) -> dict:
    """Gọi REST API với timeout bắt buộc và lỗi đã được phân loại."""
    if not PAGEINDEX_API_KEY:
        raise PageIndexUnavailable("PAGEINDEX_API_KEY chưa được cấu hình trong .env")
    try:
        response = requests.request(
            method,
            f"{BASE_URL}{path}",
            headers={"api_key": PAGEINDEX_API_KEY},
            timeout=timeout,
            **kwargs,
        )
    except requests.RequestException as error:  # timeout, DNS, TLS, mất mạng
        raise PageIndexUnavailable(f"{method} {path}: {error}") from error

    if response.status_code != 200:
        raise PageIndexAPIError(
            f"{method} {path} -> HTTP {response.status_code}: {response.text[:300]}"
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise PageIndexAPIError(f"{method} {path}: response không phải JSON") from error
    return payload if isinstance(payload, dict) else {"data": payload}


def _load_cache() -> dict:
    """Đọc mapping source -> doc_id; cache hỏng không được làm chết pipeline."""
    if not CACHE_PATH.is_file():
        return {"documents": {}}
    try:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _warn(f"Cache {CACHE_PATH.name} không đọc được — coi như chưa upload.")
        return {"documents": {}}
    documents = cache.get("documents") if isinstance(cache, dict) else None
    return {"documents": documents if isinstance(documents, dict) else {}}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def _find_font() -> Path:
    """Tìm TTF Unicode để render tiếng Việt; cho phép chỉ định qua .env."""
    configured = os.getenv("PAGEINDEX_PDF_FONT", "").strip()
    candidates = [configured, *FONT_CANDIDATES] if configured else list(FONT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Không tìm thấy font Unicode để render PDF. Đặt PAGEINDEX_PDF_FONT="
        "<đường dẫn .ttf có dấu tiếng Việt> trong .env."
    )


def _needs_char_wrap(pdf, text: str) -> bool:
    """Dòng có token dài hơn khung chữ thì mới phải cắt theo ký tự.

    Chỉ đo những token đủ dài để có khả năng tràn, nên chi phí không đáng kể so
    với việc bật wrapmode="CHAR" cho mọi dòng.
    """
    return any(
        len(token) > CHAR_WRAP_PROBE_LENGTH and pdf.get_string_width(token) > pdf.epw
        for token in text.split()
    )


def _render_pdf(document: dict, digest: str) -> Path:
    """Render Markdown chuẩn hóa thành PDF; tái dùng file cũ nếu nội dung không đổi."""
    from fpdf import FPDF

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    slug = document["id"].replace("/", "__").removesuffix(".md")
    target = PDF_DIR / f"{slug}.pdf"
    stamp = PDF_DIR / f"{slug}.sha256"
    if target.is_file() and stamp.is_file() and stamp.read_text().strip() == digest:
        return target

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("corpus", "", str(_find_font()))
    pdf.add_page()

    current_size = 0.0
    for line in document["content"].splitlines():
        stripped = line.strip()
        if not stripped:
            pdf.ln(3)
            continue

        heading = HEADING_PATTERN.match(stripped)
        if heading:
            pdf.ln(2)
            size = max(11, 17 - 2 * len(heading.group(1)))
            text = heading.group(2)
        else:
            size = 10
            text = stripped
        # set_font mỗi dòng là chi phí thừa trên corpus vài trăm nghìn ký tự.
        if size != current_size:
            pdf.set_font("corpus", size=size)
            current_size = size

        # wrapmode mặc định là WORD. Bật "CHAR" cho cả tài liệu thì fpdf2 đo lại
        # bề rộng sau từng ký tự — trên một đoạn văn dài chi phí thành bậc hai và
        # render một file 7 KB đã mất hàng phút. Chỉ URL hoặc mã văn bản dài mới
        # cần tới nó, nên bật theo từng dòng.
        # new_x="LMARGIN": mặc định multi_cell bỏ con trỏ ở mép phải ô vừa vẽ,
        # nên dòng kế tiếp còn bề rộng bằng 0 và fpdf2 ném "Not enough
        # horizontal space to render a single character".
        pdf.multi_cell(
            0,
            5,
            text,
            new_x="LMARGIN",
            new_y="NEXT",
            wrapmode="CHAR" if _needs_char_wrap(pdf, text) else "WORD",
        )

    pdf.output(str(target))
    stamp.write_text(digest, encoding="utf-8")
    return target


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    from .task4_chunking_indexing import load_documents

    if not is_enabled():
        raise PageIndexUnavailable("PAGEINDEX_API_KEY chưa được cấu hình trong .env")

    documents = load_documents()
    cache = _load_cache()
    entries: dict = cache["documents"]

    # Markdown đã bị xóa thì doc_id tương ứng không còn thuộc corpus nữa.
    live_ids = {document["id"] for document in documents}
    for stale in [key for key in entries if key not in live_ids]:
        entries.pop(stale)
        print(f"bỏ khỏi cache (không còn trong corpus): {stale}")

    for document in documents:
        digest = hashlib.sha256(document["content"].encode("utf-8")).hexdigest()
        entry = entries.get(document["id"])
        if entry and entry.get("doc_id") and entry.get("sha256") == digest:
            print(f"đã có     {document['id']} -> {entry['doc_id']}")
            continue

        pdf_path = _render_pdf(document, digest)
        with pdf_path.open("rb") as handle:
            payload = _request(
                "POST",
                "/doc/",
                files={"file": (pdf_path.name, handle, "application/pdf")},
                data={"if_retrieval": True},
                timeout=UPLOAD_TIMEOUT,
            )

        doc_id = payload.get("doc_id") or payload.get("id")
        if not isinstance(doc_id, str) or not doc_id:
            raise PageIndexAPIError(f"Response upload không có doc_id: {payload}")

        entries[document["id"]] = {
            "doc_id": doc_id,
            "sha256": digest,
            "source": document["metadata"]["source"],
            "title": document["metadata"]["title"],
            "doc_type": document["metadata"]["doc_type"],
            "url": document["metadata"]["url"],
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # Ghi ngay sau mỗi file: upload là thao tác tính phí, crash giữa chừng
        # không được làm mất doc_id đã có.
        _save_cache(cache)
        print(f"đã upload {document['id']} -> {doc_id}")

    _save_cache(cache)
    print(f"\n{len(entries)} tài liệu trong {CACHE_PATH.relative_to(ROOT_DIR)}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _first_string(node: dict, keys: tuple[str, ...]) -> str:
    """Lấy giá trị chuỗi đầu tiên có nội dung trong các key ứng viên."""
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = "\n".join(part for part in value if isinstance(part, str)).strip()
            if joined:
                return joined
    return ""


def _node_score(node: dict) -> float | None:
    for key in SCORE_KEYS:
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _extract_nodes(payload: object, depth: int = 0) -> list[dict]:
    """Tìm danh sách retrieved node trong response, không phụ thuộc một tên field."""
    if depth > 3:
        return []
    if isinstance(payload, list):
        return [
            item
            for item in payload
            if isinstance(item, dict) and _first_string(item, TEXT_KEYS)
        ]
    if not isinstance(payload, dict):
        return []
    for key in NODE_LIST_KEYS:
        if key in payload:
            nodes = _extract_nodes(payload[key], depth + 1)
            if nodes:
                return nodes
    for value in payload.values():
        if isinstance(value, (dict, list)):
            nodes = _extract_nodes(value, depth + 1)
            if nodes:
                return nodes
    return []


def _to_search_results(payload: dict, entry: dict, top_k: int) -> list[dict]:
    """Đổi retrieved nodes của một tài liệu thành SearchResult."""
    nodes = _extract_nodes(payload)[:top_k]
    if not nodes:
        return []

    # Chỉ dùng score của API khi mọi node đều có; trộn score thật với score suy
    # ra từ rank sẽ tạo ra một thang đo không có nghĩa.
    raw_scores = [_node_score(node) for node in nodes]
    use_api_scores = all(score is not None for score in raw_scores)

    results: list[dict] = []
    for index, node in enumerate(nodes):
        node_id = _first_string(node, NODE_ID_KEYS) or str(index)
        section = _first_string(node, TITLE_KEYS)
        results.append(
            {
                "id": f"pageindex::{entry['doc_id']}::{node_id}",
                "content": _first_string(node, TEXT_KEYS),
                "score": (
                    float(raw_scores[index]) if use_api_scores else 1.0 / (index + 1)
                ),
                "metadata": {
                    "source": entry["source"],
                    "title": entry["title"],
                    "doc_type": entry["doc_type"],
                    "url": entry.get("url"),
                    # Contract yêu cầu chunk_index; PageIndex không chunk theo
                    # ký tự nên dùng thứ tự node trong kết quả của tài liệu đó.
                    "chunk_index": index,
                    "section": section,
                    "section_path": section,
                    "node_id": node_id,
                },
                "retrieval_method": "pageindex",
            }
        )
    return results


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult.

    Trả list rỗng khi nhóm không bật PageIndex (thiếu key hoặc chưa upload) —
    đó là trạng thái cấu hình, không phải sự cố. Ném ``PageIndexUnavailable``
    khi dịch vụ thực sự lỗi, để Task 9 ghi nhận được và vẫn giữ hybrid result.
    """
    if not query or not query.strip() or top_k <= 0:
        return []
    if not is_enabled():
        _warn("PAGEINDEX_API_KEY trống — bỏ qua fallback, pipeline giữ hybrid result.")
        return []

    entries = [
        entry for entry in _load_cache()["documents"].values() if entry.get("doc_id")
    ]
    if not entries:
        _warn("Chưa upload tài liệu nào — chạy `python -m src.task8_pageindex_vectorless`.")
        return []

    deadline = time.monotonic() + RETRIEVAL_BUDGET_SECONDS

    # submit_query trả ngay retrieval_id, nên gửi hết rồi mới poll: độ trễ bằng
    # tài liệu chậm nhất thay vì tổng các tài liệu.
    pending: dict[str, dict] = {}
    last_error: Exception | None = None
    for entry in entries:
        if time.monotonic() >= deadline:
            break
        try:
            payload = _request(
                "POST",
                "/retrieval/",
                json={"doc_id": entry["doc_id"], "query": query, "thinking": False},
            )
        except (PageIndexUnavailable, PageIndexAPIError) as error:
            last_error = error
            _warn(f"submit query lỗi cho {entry['source']}: {error}")
            continue
        retrieval_id = payload.get("retrieval_id") or payload.get("id")
        if isinstance(retrieval_id, str) and retrieval_id:
            pending[retrieval_id] = entry

    if not pending:
        if last_error is not None:
            raise PageIndexUnavailable(f"PageIndex không nhận truy vấn nào: {last_error}")
        return []

    results: list[dict] = []
    while pending and time.monotonic() < deadline:
        for retrieval_id in list(pending):
            try:
                payload = _request("GET", f"/retrieval/{retrieval_id}/")
            except (PageIndexUnavailable, PageIndexAPIError) as error:
                _warn(f"poll retrieval lỗi: {error}")
                pending.pop(retrieval_id)
                continue

            status = str(payload.get("status", "")).lower()
            if status in FAILED_STATUSES:
                _warn(f"retrieval {retrieval_id} thất bại: {payload.get('error', status)}")
                pending.pop(retrieval_id)
            elif status in DONE_STATUSES or _extract_nodes(payload):
                results.extend(
                    _to_search_results(payload, pending.pop(retrieval_id), top_k)
                )
        if pending:
            time.sleep(min(POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))

    if pending:
        _warn(
            f"{len(pending)} retrieval chưa xong trong "
            f"{RETRIEVAL_BUDGET_SECONDS:.0f}s — trả kết quả một phần."
        )

    # Gộp nhiều tài liệu: khử trùng ID rồi sort giảm dần, phá hòa theo ID.
    unique: dict[str, dict] = {}
    for result in results:
        unique.setdefault(result["id"], result)
    ranked = sorted(unique.values(), key=lambda item: (-item["score"], item["id"]))
    return ranked[:top_k]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:])
    try:
        if question:
            print(f"Query: {question}\n")
            for rank, result in enumerate(pageindex_search(question, top_k=5), start=1):
                print(f"{rank}. [{result['score']:.4f}] {result['metadata']['title']}")
                print(f"   {result['content'][:200]}...\n".replace("\n", " ") + "\n")
        else:
            upload_documents()
    except (PageIndexUnavailable, PageIndexAPIError, FileNotFoundError) as error:
        print(f"PageIndex không dùng được: {error}", file=sys.stderr)
        print("Pipeline vẫn chạy: Task 9 giữ hybrid result khi fallback lỗi.", file=sys.stderr)
        raise SystemExit(1)
