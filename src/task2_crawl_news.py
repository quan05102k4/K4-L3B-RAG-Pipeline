"""
Task 2 — Crawl bài viết/thông báo.

Nguồn: Báo Điện tử Chính phủ (baochinhphu.vn), chuyên mục hỏi - đáp chính sách
về lao động. Đây là nguồn công khai, không chặn crawler và bám sát chủ đề thời
giờ làm việc / nghỉ ngơi / làm thêm giờ.

Ghi chú về nguồn đã loại: thuvienphapluat.vn trả về Cloudflare JS challenge
("Blocked by anti-bot protection") với cả requests lẫn Playwright. Theo yêu cầu
của đề bài, không tìm cách vượt cơ chế chặn — thay bằng URL dễ crawl hơn.

Mỗi bài được lưu thành một JSON trong data/landing/news/ với tối thiểu:
    url, title, date_crawled, content_markdown

Cài browser trước khi chạy:
    python -m playwright install chromium

Chạy:
    python -m src.task2_crawl_news
"""

import asyncio
import json
import re
from datetime import date
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # Thời giờ làm việc bình thường, làm việc ban đêm.
    "https://baochinhphu.vn/thoi-gian-lam-viec-cua-nguoi-lao-dong-quy-dinh-the-nao-102224781.htm",
    # Điều kiện và giới hạn làm thêm giờ.
    "https://baochinhphu.vn/lam-them-gio-the-nao-la-dung-quy-dinh-102240130085419703.htm",
    # Cách tính tiền lương làm thêm giờ và làm việc ban đêm.
    "https://baochinhphu.vn/cach-tinh-tien-luong-lam-ca-dem-va-them-gio-102260874.htm",
    # Nghỉ trong giờ làm việc, nghỉ giữa ca.
    "https://baochinhphu.vn/thoi-gian-nghi-trong-gio-lam-viec-co-duoc-tinh-tra-luong-102291050.htm",
    # Cách tính số ngày nghỉ hằng năm (phép năm).
    "https://baochinhphu.vn/huong-dan-tinh-so-ngay-nghi-hang-nam-cua-nguoi-lao-dong-102257923.htm",
    # Các loại hợp đồng lao động.
    "https://baochinhphu.vn/the-nao-la-hop-dong-lao-dong-khong-xac-dinh-thoi-han-102211150.htm",
]

# Vùng nội dung chính của baochinhphu.vn; mọi thứ ngoài vùng này (menu, quảng
# cáo, tin liên quan, footer) không được đưa vào Markdown. Title nằm ở field
# riêng của JSON nên không lấy thẻ h1 để tránh lặp tiêu đề.
CONTENT_SELECTORS = ["div.detail-sapo", "div.detail-content"]

BOILERPLATE_PATTERNS = [
    # Dòng disclaimer cuối chuyên mục hỏi - đáp, không mang thông tin quy định.
    re.compile(r"^[_*\s]*\*\s*Thông tin chuyên mục có giá trị tham khảo", re.IGNORECASE),
    re.compile(r"^!\[.*$"),  # dòng chỉ chứa ảnh
    re.compile(r"^_{2,}$"),
    re.compile(r"^\s*\*{2,}\s*$"),
]

MIN_CONTENT_CHARS = 500


def clean_markdown(markdown: str) -> str:
    """Bỏ dòng rác còn sót và chuẩn hóa khoảng trắng."""
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if any(pattern.match(stripped) for pattern in BOILERPLATE_PATTERNS):
            continue
        # Dòng chỉ còn dấu nhấn rỗng kiểu "****" sau khi bỏ thẻ.
        if stripped and not stripped.strip("*_ "):
            continue
        # Chuỗi dấu * dư thừa do thẻ <strong> lồng nhau, ví dụ "**Chinhphu.vn******".
        line = re.sub(r"\*{3,}", "**", line)
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def crawl_article(url: str) -> dict:
    """Crawl một bài viết và trả về dictionary đúng schema bắt buộc."""
    browser = BrowserConfig(headless=True, verbose=False)
    config = CrawlerRunConfig(
        target_elements=CONTENT_SELECTORS,
        excluded_tags=["script", "style", "nav", "footer", "form", "aside", "iframe"],
        exclude_external_links=True,
        exclude_social_media_links=True,
        cache_mode=CacheMode.BYPASS,
        page_timeout=60_000,
    )

    async with AsyncWebCrawler(config=browser) as crawler:
        result = await crawler.arun(url=url, config=config)

    if not result.success:
        raise RuntimeError(result.error_message or "crawl failed")

    metadata = result.metadata or {}
    title = (metadata.get("title") or metadata.get("og:title") or "").strip()
    if not title:
        raise RuntimeError("không lấy được title")

    markdown = result.markdown
    raw = getattr(markdown, "raw_markdown", None) or str(markdown)
    content = clean_markdown(raw)

    # Sapo (mô tả đầu bài) mang thông tin tóm tắt quy định, ghép vào đầu nội dung
    # nếu trang không render sapo trong vùng nội dung.
    summary = (metadata.get("description") or "").strip()
    if summary and summary[:60] not in content:
        content = f"{summary}\n\n{content}"

    if len(content) < MIN_CONTENT_CHARS:
        raise RuntimeError(f"nội dung quá ngắn ({len(content)} ký tự), có thể bị chặn")

    return {
        "url": url,
        "title": title,
        "date_crawled": date.today().isoformat(),
        "content_markdown": content,
        # Provenance bổ sung: truy ngược về nguồn công khai và phiên bản bài viết.
        "source_name": "Báo Điện tử Chính phủ",
        "source_domain": "baochinhphu.vn",
        "date_published": metadata.get("article:published_time"),
        "date_modified": metadata.get("article:modified_time"),
        "crawler": "crawl4ai",
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved += 1
            print(f"Saved: {output.name} ({len(article['content_markdown']):,} chars)")
        except Exception as error:
            print(f"Failed: {url} — {error}")

    print(f"Done: {saved}/{len(ARTICLE_URLS)} articles in {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
