"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Chủ đề của nhóm: quy định về thời giờ làm việc, thời giờ nghỉ ngơi và làm thêm
giờ của người lao động tại Việt Nam.

Nguồn: Công báo điện tử nước CHXHCN Việt Nam (congbao.chinhphu.vn). Bản PDF
đăng Công báo là bản chế bản có lớp text nên chuẩn hóa sang Markdown được;
bản "signed" trên datafiles.chinhphu.vn là bản scan chữ ký nên không trích xuất
được text (đã kiểm tra: 0 ký tự). Mỗi văn bản vẫn lưu kèm ``official_page_url``
trỏ về trang văn bản của Cổng thông tin điện tử Chính phủ để truy vết.

Script sẽ đọc trang Công báo của từng văn bản để lấy link PDF hiện hành, tải về
``data/landing/legal/`` và ghi manifest ``sources.json`` (URL, ngày tải, số
bytes, sha256) phục vụ provenance.

Văn bản dài được Công báo đăng làm nhiều phần (ví dụ Nghị định 145/2020/NĐ-CP
đăng ở số 1203+1204 và 1205+1206). Mỗi phần là một file ``*_phan_N.pdf`` và
được Task 3 ghép lại thành một Markdown duy nhất.

Chạy:
    python -m src.task1_collect_legal_docs
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import requests


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "sources.json"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = (15, 180)  # (connect, read) — CDN Công báo có lúc phản hồi chậm.
MAX_ATTEMPTS = 3
MIN_PDF_BYTES = 50 * 1024

GAZETTE_PDF_PATTERN = re.compile(
    r"""href=["'](https://congbaocdn\.chinhphu\.vn/[^"']+\.pdf)["']""", re.IGNORECASE
)

# Tối thiểu 3 văn bản; văn bản thứ 4 mở rộng độ phủ sang an toàn, vệ sinh lao động.
LEGAL_SOURCES = [
    {
        "slug": "bo_luat_lao_dong_2019",
        "title": "Bộ luật Lao động",
        "document_number": "45/2019/QH14",
        "document_type": "Bộ luật",
        "issuing_body": "Quốc hội",
        "issued_date": "2019-11-20",
        "effective_date": "2021-01-01",
        "gazette_page_url": "https://congbao.chinhphu.vn/van-ban/nghi-quyet-so-45-2019-qh14-30232.htm",
        "official_page_url": "https://vanban.chinhphu.vn/?docid=198540&pageid=27160",
        "expected_parts": 1,
        "role_in_corpus": (
            "Văn bản gốc quy định thời giờ làm việc, thời giờ nghỉ ngơi và làm "
            "thêm giờ (Điều 98, 105-113) cùng các quy định về hợp đồng lao động."
        ),
    },
    {
        "slug": "nghi_dinh_145_2020",
        "title": (
            "Nghị định quy định chi tiết và hướng dẫn thi hành một số điều của "
            "Bộ luật Lao động về điều kiện lao động và quan hệ lao động"
        ),
        "document_number": "145/2020/NĐ-CP",
        "document_type": "Nghị định",
        "issuing_body": "Chính phủ",
        "issued_date": "2020-12-14",
        "effective_date": "2021-02-01",
        "gazette_page_url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-145-2020-nd-cp-32732.htm",
        "official_page_url": "https://chinhphu.vn/default.aspx?docid=201967&pageid=27160",
        "expected_parts": 2,
        "role_in_corpus": (
            "Hướng dẫn chi tiết cách tính tiền lương làm thêm giờ, làm việc ban "
            "đêm và thời giờ được tính vào giờ làm việc có hưởng lương."
        ),
        "note": (
            "Cơ sở dữ liệu quốc gia ghi nhận văn bản đã được sửa đổi, bổ sung "
            "một phần. Giữ nguyên bản Công báo kèm ngày tải để có thể truy vết "
            "phiên bản đã dùng."
        ),
    },
    {
        "slug": "nghi_dinh_12_2022",
        "title": (
            "Nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực lao "
            "động, bảo hiểm xã hội, người lao động Việt Nam đi làm việc ở nước "
            "ngoài theo hợp đồng"
        ),
        "document_number": "12/2022/NĐ-CP",
        "document_type": "Nghị định",
        "issuing_body": "Chính phủ",
        "issued_date": "2022-01-17",
        "effective_date": "2022-01-17",
        "gazette_page_url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-12-2022-nd-cp-36716.htm",
        "official_page_url": "https://vanban.chinhphu.vn/?classid=1&docid=205182&pageid=27160",
        "expected_parts": 1,
        "role_in_corpus": (
            "Bổ sung góc nhìn xử phạt khi vi phạm quy định về thời giờ làm việc, "
            "thời giờ nghỉ ngơi và hợp đồng lao động."
        ),
    },
    {
        "slug": "nghi_quyet_17_2022",
        "title": (
            "Nghị quyết về số giờ làm thêm trong 01 năm, trong 01 tháng của "
            "người lao động trong bối cảnh phòng, chống dịch COVID-19 và phục "
            "hồi, phát triển kinh tế - xã hội"
        ),
        "document_number": "17/2022/UBTVQH15",
        "document_type": "Nghị quyết",
        "issuing_body": "Ủy ban Thường vụ Quốc hội",
        "issued_date": "2022-03-23",
        "effective_date": "2022-04-01",
        "gazette_page_url": "https://congbao.chinhphu.vn/van-ban/nghi-quyet-so-17-2022-ubtvqh15-36956.htm",
        "official_page_url": "https://congbao.chinhphu.vn/noi-dung-van-ban-so-17-2022-ubtvqh15-36956?cbid=40162",
        "expected_parts": 1,
        "role_in_corpus": (
            "Nâng trần số giờ làm thêm (tối đa 60 giờ/tháng, 300 giờ/năm) so "
            "với Điều 107 Bộ luật Lao động — nguồn đối chiếu cho câu hỏi về "
            "giới hạn làm thêm giờ."
        ),
    },
]


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"})
    return session


def _get(session: requests.Session, url: str) -> requests.Response:
    """GET có retry — CDN Công báo thỉnh thoảng timeout ở lần kết nối đầu."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.RequestException as error:  # pragma: no cover - phụ thuộc mạng
            last_error = error
            print(f"    retry {attempt}/{MAX_ATTEMPTS}: {type(error).__name__}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Cannot fetch {url}: {last_error}")


def resolve_pdf_urls(session: requests.Session, source: dict) -> list[str]:
    """Đọc trang Công báo để lấy link PDF của văn bản (theo đúng thứ tự phần)."""
    page = _get(session, source["gazette_page_url"])
    urls = list(dict.fromkeys(GAZETTE_PDF_PATTERN.findall(page.text)))
    if not urls:
        raise RuntimeError(
            f"{source['document_number']}: không tìm thấy link PDF trên "
            f"{source['gazette_page_url']}"
        )
    if len(urls) != source["expected_parts"]:
        print(
            f"    ! Trang Công báo có {len(urls)} phần, khác với "
            f"expected_parts={source['expected_parts']} — kiểm tra lại nguồn."
        )
    return [urljoin(page.url, url) for url in urls]


def part_filename(source: dict, index: int, total: int) -> str:
    """Một phần -> <slug>.pdf; nhiều phần -> <slug>_phan_N.pdf."""
    if total == 1:
        return f"{source['slug']}.pdf"
    return f"{source['slug']}_phan_{index}.pdf"


def download_documents() -> list[dict]:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai và ghi manifest provenance."""
    session = _session()
    manifest: list[dict] = []

    for source in LEGAL_SOURCES:
        print(f"- {source['document_number']} — {source['title'][:60]}")
        pdf_urls = resolve_pdf_urls(session, source)
        parts: list[dict] = []

        for index, pdf_url in enumerate(pdf_urls, 1):
            response = _get(session, pdf_url)
            content = response.content
            if not content.startswith(b"%PDF") or len(content) < MIN_PDF_BYTES:
                raise RuntimeError(
                    f"{source['document_number']} phần {index}: nội dung tải về "
                    f"không phải PDF hợp lệ ({len(content):,} bytes)."
                )

            # Ghi đè đúng file tương ứng, không tạo bản sao khi chạy lại.
            target = DATA_DIR / part_filename(source, index, len(pdf_urls))
            target.write_bytes(content)
            print(f"    saved {target.name} ({len(content):,} bytes)")

            parts.append(
                {
                    "part": index,
                    "filename": target.name,
                    "landing_path": f"data/landing/legal/{target.name}",
                    "file_url": pdf_url,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )

        record = {key: value for key, value in source.items() if key != "expected_parts"}
        record["date_downloaded"] = date.today().isoformat()
        record["parts"] = parts
        manifest.append(record)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    setup_directory()
    documents = download_documents()
    files = sum(len(item["parts"]) for item in documents)
    print(f"Done: {len(documents)} legal documents ({files} files) in {DATA_DIR}")
