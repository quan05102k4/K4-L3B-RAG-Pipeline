# Individual contribution report

## Thông tin

- Họ và tên: Nguyễn Minh Quân
- Mã học viên: 2A202602490
- Nhóm: K4-L3B — chatbot RAG pháp luật lao động Việt Nam
- Repository/branch: https://github.com/quan05102k4/K4-L3B-RAG-Pipeline — `main`
- Vai trò: Trưởng nhóm — Data và Retrieval

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 1 — thu thập văn bản pháp luật | Chọn 4 văn bản (Bộ luật Lao động 45/2019/QH14, NĐ 145/2020, NĐ 12/2022, NQ 17/2022) từ Công báo; viết downloader ghi manifest provenance (`url`, `date_downloaded`, `bytes`, `sha256`); xử lý văn bản đăng nhiều kỳ (NĐ 145/2020 = 2 PDF) | `src/task1_collect_legal_docs.py`, `data/landing/legal/sources.json`, 5 PDF trong `data/landing/legal/` | Done |
| Task 2 — crawl bài viết | Crawl 6 bài chuyên mục hỏi–đáp chính sách lao động của baochinhphu.vn bằng Crawl4AI; giới hạn vùng nội dung theo `CONTENT_SELECTORS`, lọc boilerplate | `src/task2_crawl_news.py`, `data/landing/news/article_01..06.json` | Done |
| Task 3 — chuẩn hoá Markdown | Viết `normalize_legal_text()`: bỏ tiêu đề trang Công báo và số trang, nối dòng bị ngắt theo layout in (kể cả ngắt qua trang), giữ ranh giới Chương/Mục/Điều thành heading; ghép 2 phần NĐ 145/2020 thành một file | `src/task3_convert_markdown.py`, 10 file trong `data/standardized/` (868 KB) | Done |
| Task 4 — chunking, embedding, index | Chunker structure-aware theo heading + recursive split, breadcrumb trong `content`, ID ổn định `<path>::chunk-<n>`, upsert + dọn chunk mồ côi; `embed_texts()` dùng chung cho corpus và query | `src/task4_chunking_indexing.py` — 1180 chunk, bge-m3 1024 chiều | Done |
| Task 5/6 — dense + BM25 | Hai nhánh cùng trả `SearchResult` một schema; dense đọc cosine từ ChromaDB, BM25 chạy trên đúng corpus chunks | `src/task5_semantic_search.py`, `src/task6_lexical_search.py` | Done |
| Task 7 — RRF | Fuse theo thứ hạng, khử trùng ID trong từng danh sách, copy dict trước khi ghi đè score để không phá cosine gốc của dense | `src/task7_reranking.py` | Done |
| Task 9 — retrieval pipeline + fallback | RRF chạy đúng một lần; fallback đọc cosine gốc chứ không đọc RRF score; provider lỗi thì giữ hybrid thay vì crash | `src/task9_retrieval_pipeline.py` | Done |
| Hiệu chỉnh `SCORE_THRESHOLD` | Đo 5 query trong domain và 5 query ngoài domain, chọn 0.60 | `group_project/evaluation/RESULT.md` § "Hiệu chỉnh fallback threshold" | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Nâng `CHUNK_SIZE` từ 500 (starter đề xuất) lên 1000, overlap 150, và chunk theo heading trước khi cắt.
   **Lý do/evidence:** Tôi đo phân bố độ dài thật của corpus: 421 khối "Điều" trong 4 văn bản có median 902 ký tự, p25 488. Ở 500 ký tự phần lớn Điều bị cắt làm 2–4 mảnh nên retrieval trả về một khoản mất đầu; ở 1000 ký tự quá nửa số Điều nằm trọn trong một chunk. Breadcrumb "<văn bản> > Chương … > Điều …" được gắn vào `content` nên chunk giữa Điều 105 vẫn đủ ngữ cảnh cho cả dense lẫn BM25.
   **Trade-off:** Chunk dài hơn làm context precision giảm (mỗi chunk mang thêm chữ không liên quan) và vẫn chưa đủ: case q05 trong evaluation cho thấy khoản 3 Điều 107 dài hơn 1000 ký tự vẫn bị cắt mất điểm d, đ → context recall 0.00 dù bằng chứng ở rank 1.

2. **Quyết định:** Nhánh quyết định fallback đọc **cosine score gốc của dense**, không đọc RRF score — dù RRF là bước chạy ngay trước đó.
   **Lý do/evidence:** Tôi đo trực tiếp hai query: "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?" (trong domain, cosine 0.7476) và "Tỷ số trận chung kết World Cup 2022 là bao nhiêu?" (ngoài domain, cosine 0.3923) cho **RRF score của top-1 bằng nhau đến từng chữ số: 0.032522**. RRF chỉ đọc vị trí trong danh sách nên chunk đứng đầu cả hai nhánh luôn nhận `1/61 + 1/62`, bất kể có liên quan hay không. Không tồn tại ngưỡng nào trên RRF score phân biệt được hai query đó.
   **Trade-off:** Ngưỡng 0.60 chỉ đúng cho corpus lao động + bge-m3. Đổi corpus hoặc đổi embedding model là phải đo lại; tôi đưa `SCORE_THRESHOLD` ra `.env` và ghi cách đo vào RESULT.md thay vì hard-code.

## Kiểm thử và kết quả

- **Test khoá hành vi phần tôi làm:** `test_chunk_documents_preserves_identity_and_metadata`, `test_semantic_search_uses_shared_embedding_and_contract`, `test_lexical_search_returns_bm25_contract`, `test_rrf_uses_rank_deduplicates_and_marks_hybrid`, `test_retrieve_uses_dense_score_for_fallback`, `test_retrieve_fuses_once_when_dense_is_confident`, `test_retrieve_survives_fallback_provider_error` — tất cả pass trong `pytest -q` (20/20).
- **Hiệu chỉnh threshold:** 5 query trong domain cho best dense cosine 0.7329–0.8022; 5 query ngoài domain cho 0.3523–0.4762. Khoảng trống 0.4762–0.7329, chọn 0.60 ≈ điểm giữa, cách mỗi biên ~0.13. Ca sát biên nhất là "học phí đại học ngành CNTT" (0.4762) vì vẫn là ngôn ngữ chính sách.
- **Lỗi đã phát hiện và cách xử lý:**
  - Nguồn `thuvienphapluat.vn` trả Cloudflare JS challenge với cả `requests` lẫn Playwright. Theo yêu cầu đề bài, tôi **không** tìm cách vượt cơ chế chặn mà đổi sang baochinhphu.vn, và ghi lý do loại nguồn vào docstring Task 2.
  - Bản PDF "signed" trên datafiles.chinhphu.vn là bản scan, trích ra 0 ký tự. Tôi chuyển sang bản chế bản của Công báo (có lớp text) và vẫn lưu `official_page_url` để truy vết.
  - Evaluation lộ ra lỗi **RRF cộng điểm kép** ở q18: BM25 top-10 có 4/10 chunk cùng thuộc `news/article_04.md` (tiêu đề bài báo trùng gần nguyên văn câu hỏi), đẩy bằng chứng chỉ-dense (Điều 58, rank 4) xuống hạng 6 — thua chunk trên nó **0.000248 điểm**. Đây là lỗi thiết kế trong phần tôi phụ trách. Bản vá tôi đề xuất và ưu tiên số 1 trong RESULT.md: giới hạn số chunk mỗi `doc_id` trước khi đưa vào `rerank_rrf` (tối đa 2 chunk/tài liệu), kiểm chứng được bằng `--retrieval-only` mà không tốn token LLM.

## Điều còn hạn chế

- **Hạn chế cụ thể:** Chunker cắt cứng theo `CHUNK_SIZE` nên một danh sách điều khoản phải đọc trọn mới trả lời đúng vẫn bị đứt (q05: `chunk-166` dài 765 ký tự nhưng thiếu điểm d, đ của khoản 3 Điều 107). Config A chỉ tránh được lỗi này nhờ tình cờ vớ thêm một nguồn trùng lặp (NĐ 145/2020 Điều 61), không phải nhờ thiết kế.
- **Nếu có thêm thời gian:** Đổi đơn vị chunk từ "1000 ký tự" sang "một Điều/khoản", cho phép chunk vượt `CHUNK_SIZE` khi cần để không cắt giữa một danh sách — rồi reindex và so lại `context_recall` của q05 ở cả hai config. Lưu ý reindex làm đổi corpus SHA-256 nên số mới chỉ so được với baseline chạy lại trên cùng index.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Nguyễn Minh Quân
