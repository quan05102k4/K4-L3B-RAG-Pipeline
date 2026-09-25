"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Legal:
    PDF Công báo -> MarkItDown -> chuẩn hóa cấu trúc -> data/standardized/legal/

    Text trích từ PDF bị ngắt dòng theo layout in (căn đều hai bên), lẫn tiêu đề
    trang "CÔNG BÁO/Số .../Ngày ..." và số trang. Hàm normalize_legal_text() xử lý:
      - bỏ tiêu đề trang, số trang, ghi chú "(Xem tiếp theo Công báo số ...)";
      - nối lại các dòng bị ngắt giữa câu (kể cả ngắt qua trang);
      - giữ nguyên ranh giới Chương / Mục / Điều / khoản / điểm và đánh dấu
        Chương, Mục, Điều thành heading Markdown.

    Văn bản được Công báo đăng nhiều kỳ (Nghị định 145/2020/NĐ-CP) được ghép
    theo thứ tự phần thành một file Markdown duy nhất.

News:
    JSON landing -> Markdown có header metadata -> data/standardized/news/

Chạy:
    python -m src.task3_convert_markdown
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from markitdown import MarkItDown


ROOT_DIR = Path(__file__).parent.parent
LANDING_DIR = ROOT_DIR / "data" / "landing"
OUTPUT_DIR = ROOT_DIR / "data" / "standardized"
LEGAL_MANIFEST = LANDING_DIR / "legal" / "sources.json"

# --- Dòng rác sinh ra từ layout của bản in Công báo ------------------------
NOISE_PATTERNS = [
    re.compile(r"^CÔNG BÁO\s*/\s*Số.*$", re.IGNORECASE),
    re.compile(r"^\d{1,4}\s*CÔNG BÁO.*$", re.IGNORECASE),
    re.compile(r"^\(?\s*(Xem\s+)?tiếp theo Công báo số.*$", re.IGNORECASE),
    re.compile(r"^VĂN BẢN QUY PHẠM PHÁP LUẬT$", re.IGNORECASE),
    re.compile(r"^\d{1,4}$"),  # số trang đứng một mình
    re.compile(r"^[\s\.\-_]*$"),
]

# --- Ranh giới cấu trúc pháp lý -------------------------------------------
CHUONG_PATTERN = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\s*$")
MUC_PATTERN = re.compile(r"^Mục\s+(\d+)\s*$")
PHU_LUC_PATTERN = re.compile(r"^Phụ lục\s+([IVXLCDM]+|\d+)\s*$")
# Yêu cầu dấu chấm ngay sau số để không bắt nhầm câu dẫn chiếu
# ("... quy định tại Điều 185 của Bộ luật Lao động").
DIEU_PATTERN = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
KHOAN_PATTERN = re.compile(r"^\d+\.\s+\S")
DIEM_PATTERN = re.compile(r"^[a-zđáâăêôơưA-ZĐ]{1,2}\)\s+\S")

STRUCTURE_PATTERNS = [
    CHUONG_PATTERN,
    MUC_PATTERN,
    PHU_LUC_PATTERN,
    DIEU_PATTERN,
    KHOAN_PATTERN,
    DIEM_PATTERN,
]

# Tiêu đề Điều coi như đã trọn vẹn khi kết thúc bằng một trong các ký tự này.
HEADING_STOP_CHARS = (".", ":", ";", "!", "?")
# Chỉ tách đoạn ở ranh giới câu thật sự — dấu ";" thường nối hai vế của cùng
# một khoản nên không dùng để tách.
PARAGRAPH_END_CHARS = (".", "!", "?")
# Dòng tiêu đề Điều bị ngắt khi dòng đã chạm mép phải của bản in.
WRAPPED_LINE_MIN_CHARS = 65
MIN_MARKDOWN_CHARS = 200


def _is_noise(line: str) -> bool:
    return any(pattern.match(line) for pattern in NOISE_PATTERNS)


def _starts_structure(line: str) -> bool:
    return any(pattern.match(line) for pattern in STRUCTURE_PATTERNS)


def _read_lines(text: str) -> list[tuple[str, bool]]:
    """Chuẩn hóa khoảng trắng và đánh dấu dòng nào có dòng trống đứng trước.

    Dòng trống là tín hiệu ngắt đoạn của pdfminer — có file giữ được, có file
    không, nên chỉ dùng như tín hiệu phụ bên cạnh ranh giới cấu trúc.
    """
    result: list[tuple[str, bool]] = []
    blank_before = False
    for raw_line in text.replace("\x0c", "\n").splitlines():
        line = re.sub(r"[ \t ]+", " ", raw_line).strip()
        if not line or _is_noise(line):
            blank_before = True
            continue
        result.append((line, blank_before))
        blank_before = False
    return result


def _heading_title(lines: list[tuple[str, bool]], index: int, title: str) -> tuple[str, int]:
    """Nối phần tiêu đề Điều bị ngắt xuống dòng dưới.

    Chỉ nối khi dòng hiện tại đã chạm mép phải (bị wrap) và dòng kế tiếp không
    mở đầu một đơn vị cấu trúc mới — nhờ vậy "Điều 106. Giờ làm việc ban đêm"
    không bị nuốt mất đoạn nội dung ngay sau nó.
    """
    cursor = index
    current = title
    while (
        len(lines[cursor][0]) >= WRAPPED_LINE_MIN_CHARS
        and not current.endswith(HEADING_STOP_CHARS)
        and cursor + 1 < len(lines)
        and not _starts_structure(lines[cursor + 1][0])
        and len(lines[cursor + 1][0]) < len(lines[cursor][0])
    ):
        cursor += 1
        current = f"{current} {lines[cursor][0]}".strip()
    return current, cursor


def _strip_continuation_banner(
    lines: list[tuple[str, bool]], document_number: str
) -> list[tuple[str, bool]]:
    """Bỏ banner lặp lại ở đầu phần đăng tiếp của Công báo.

    Kỳ đăng tiếp mở đầu bằng tên cơ quan và tên văn bản (chứa số hiệu) trước khi
    vào nội dung. Chỉ cắt khi đoạn mở đầu đó thật sự chứa số hiệu văn bản, tránh
    cắt nhầm phần nội dung bị ngắt giữa chừng giữa hai kỳ đăng.
    """
    head: list[str] = []
    cut = 0
    for position, (line, _) in enumerate(lines):
        if _starts_structure(line):
            cut = position
            break
        head.append(line)
    else:
        return lines

    return lines[cut:] if document_number in " ".join(head) else lines


def normalize_legal_text(
    text: str, document_number: str = "", is_continuation: bool = False
) -> str:
    """Biến text thô của PDF Công báo thành Markdown giữ cấu trúc Điều/Khoản/Điểm."""
    lines = _read_lines(text)
    if is_continuation and document_number:
        lines = _strip_continuation_banner(lines, document_number)
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(" ".join(paragraph).strip())
            paragraph.clear()

    index = 0
    while index < len(lines):
        line, blank_before = lines[index]

        chuong = CHUONG_PATTERN.match(line)
        muc = MUC_PATTERN.match(line)
        phu_luc = PHU_LUC_PATTERN.match(line)
        dieu = DIEU_PATTERN.match(line)

        if chuong or muc or phu_luc:
            flush()
            # Tên chương/mục/phụ lục nằm ở dòng kế tiếp, thường viết hoa toàn bộ.
            name = ""
            if index + 1 < len(lines) and not _starts_structure(lines[index + 1][0]):
                name = lines[index + 1][0]
                index += 1
            level = "##" if (chuong or phu_luc) else "###"
            blocks.append(f"{level} {line}{'. ' + name if name else ''}")
        elif dieu:
            flush()
            title, index = _heading_title(lines, index, dieu.group(2).strip())
            heading = f"Điều {dieu.group(1)}." + (f" {title}" if title else "")
            blocks.append(f"#### {heading}")
        elif KHOAN_PATTERN.match(line) or DIEM_PATTERN.match(line):
            flush()
            paragraph.append(line)
        elif (
            paragraph
            and blank_before
            and line[:1].isupper()
            and " ".join(paragraph).rstrip().endswith(PARAGRAPH_END_CHARS)
        ):
            # Đoạn không đánh số nằm trong cùng một khoản (ví dụ đoạn "Nhà nước
            # khuyến khích..." của Điều 105) — tách đoạn khi nguồn có dòng trống.
            flush()
            paragraph.append(line)
        else:
            paragraph.append(line)

        index += 1

    flush()
    return "\n\n".join(block for block in blocks if block)


def _legal_header(document: dict) -> str:
    """Header metadata để mỗi Markdown tự truy ngược được về nguồn."""
    parts = document["parts"]
    landing = ", ".join(part["landing_path"] for part in parts)
    rows = [
        f"# {document['document_type']} số {document['document_number']} — {document['title']}",
        "",
        f"- **Số hiệu:** {document['document_number']}",
        f"- **Loại văn bản:** {document['document_type']}",
        f"- **Cơ quan ban hành:** {document['issuing_body']}",
        f"- **Ngày ban hành:** {document['issued_date']}",
        f"- **Ngày có hiệu lực:** {document['effective_date']}",
        f"- **Nguồn công khai:** {document['gazette_page_url']}",
        f"- **Trang văn bản chính thức:** {document['official_page_url']}",
        f"- **File gốc:** {landing}",
        f"- **Ngày thu thập:** {document['date_downloaded']}",
    ]
    if document.get("note"):
        rows.append(f"- **Lưu ý:** {document['note']}")
    rows += ["", "---", "", ""]
    return "\n".join(rows)


def convert_legal_docs() -> int:
    """Convert PDF Công báo trong landing/legal sang standardized/legal."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LEGAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Thiếu {LEGAL_MANIFEST}. Chạy `python -m src.task1_collect_legal_docs` trước."
        )

    documents = json.loads(LEGAL_MANIFEST.read_text(encoding="utf-8"))
    converter = MarkItDown()
    converted = 0

    for document in documents:
        sections: list[str] = []
        for part in sorted(document["parts"], key=lambda item: item["part"]):
            source = legal_dir / part["filename"]
            if not source.exists():
                raise FileNotFoundError(f"Thiếu file landing: {source}")
            raw_text = converter.convert(str(source)).text_content
            if not raw_text.strip():
                raise RuntimeError(
                    f"{source.name}: không trích xuất được text (PDF scan?). "
                    "Cần đổi sang bản có lớp text."
                )
            sections.append(
                normalize_legal_text(
                    raw_text,
                    document_number=document["document_number"],
                    is_continuation=part["part"] > 1,
                )
            )

        body = "\n\n".join(sections)
        markdown = _legal_header(document) + body + "\n"
        # Ghi đè đúng file tương ứng khi chạy lại, không tạo bản sao.
        target = output_dir / f"{document['slug']}.md"
        target.write_text(markdown, encoding="utf-8")
        converted += 1
        print(f"Legal: {target.name} ({len(markdown):,} chars, {len(sections)} part(s))")

    return converted


def convert_news_articles() -> int:
    """Convert JSON trong landing/news sang standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = {"url", "title", "date_crawled", "content_markdown"} - data.keys()
        if missing:
            raise ValueError(f"{path.name} thiếu trường bắt buộc: {sorted(missing)}")

        rows = [
            f"# {data['title']}",
            "",
            f"- **Nguồn:** {data['url']}",
            f"- **Trang nguồn:** {data.get('source_name', '')}".rstrip(),
            f"- **Ngày đăng:** {data.get('date_published') or 'không rõ'}",
            f"- **Ngày thu thập:** {data['date_crawled']}",
            f"- **File gốc:** data/landing/news/{path.name}",
            "",
            "---",
            "",
        ]
        markdown = "\n".join(rows) + data["content_markdown"].strip() + "\n"

        if len(markdown.strip()) < MIN_MARKDOWN_CHARS:
            raise ValueError(f"{path.name}: nội dung chuẩn hóa quá ngắn.")

        target = output_dir / f"{path.stem}.md"
        target.write_text(markdown, encoding="utf-8")
        converted += 1
        print(f"News: {target.name} ({len(markdown):,} chars)")

    return converted


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legal = convert_legal_docs()
    news = convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR} ({legal} legal, {news} news)")


if __name__ == "__main__":
    convert_all()
