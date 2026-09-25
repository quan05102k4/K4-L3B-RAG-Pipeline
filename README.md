# K4-L3B — Chatbot RAG pháp luật lao động Việt Nam

Chatbot hỏi đáp về **thời giờ làm việc, thời giờ nghỉ ngơi và làm thêm giờ** của
người lao động tại Việt Nam. Mỗi câu trả lời chỉ dựa trên tài liệu nhóm đã thu
thập và trích dẫn theo nhãn `[Document N]` trỏ đúng về chunk đã đưa vào context;
câu hỏi ngoài phạm vi tài liệu nhận được câu từ chối thay vì phỏng đoán.

Thành viên và phân công: [TEAMMATES.md](TEAMMATES.md).

## Corpus

| Loại | Số lượng | Nguồn |
| ---- | -------: | ----- |
| Văn bản pháp luật | 4 (5 file PDF) | Công báo điện tử — congbao.chinhphu.vn |
| Bài viết | 6 | Báo Điện tử Chính phủ — baochinhphu.vn, chuyên mục hỏi–đáp chính sách lao động |

Văn bản pháp luật: Bộ luật Lao động 45/2019/QH14, Nghị định 145/2020/NĐ-CP (đăng
Công báo làm 2 phần), Nghị định 12/2022/NĐ-CP, Nghị quyết 17/2022/UBTVQH15.

Provenance của từng file (URL, ngày tải, số bytes, `sha256`, trang văn bản chính
thức) nằm trong [`data/landing/legal/sources.json`](data/landing/legal/sources.json).
Bản Markdown đã chuẩn hoá nằm trong `data/standardized/` — 10 file, 868 KB, index
thành **1180 chunk**.

## Kiến trúc

```
data/landing/          Task 1 (PDF Công báo) + Task 2 (JSON bài báo, Crawl4AI)
      |
data/standardized/     Task 3 — MarkItDown + normalize_legal_text()
      |                          giữ ranh giới Chương / Mục / Điều
chroma_db/             Task 4 — chunk 1000/150 theo heading, breadcrumb,
      |                          embed bge-m3 (1024 chiều), upsert vào ChromaDB
      +-- Task 5 dense (cosine)  --+
      |                            +-- Task 7 RRF (k=60) -- Task 9 pipeline
      +-- Task 6 BM25            --+                             |
                                    Task 8 PageIndex fallback  <--+ (khi dense
      |                                                             dưới ngưỡng)
Task 10 generation có citation  ->  app.py (Streamlit)
```

Bốn invariant mà code và test cùng giữ:

- Dense và BM25 trả **cùng một schema** `SearchResult` (sort giảm dần, ID duy nhất).
- **RRF chỉ chạy một lần** trong toàn pipeline và chỉ gộp *thứ hạng*, không gộp score.
- **Fallback đọc cosine score gốc của dense**, không đọc RRF score — RRF score của
  một query trong domain và một query ngoài domain có thể bằng nhau đến từng chữ số.
- Nhãn `[Document N]` trong answer luôn trỏ đúng `sources[N-1]`, kể cả sau khi
  `reorder_for_llm()` đổi vị trí chunk trong context.

Chi tiết schema và interface: [docs/MODULE_CONTRACTS.md](docs/MODULE_CONTRACTS.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Điền API key vào `.env` (nhóm chạy `LLM_PROVIDER=groq` + `GROQ_API_KEY`); file này
không được commit. Xem [.env.example](.env.example) cho danh sách biến.

### Chạy lại từ đầu

`data/landing/` và `data/standardized/` đã có sẵn trong repo, nên bước tối thiểu
để chạy sản phẩm là **dựng lại index** (`chroma_db/` không commit vì tái tạo được):

```bash
python -m src.task4_chunking_indexing     # bắt buộc — dựng chroma_db/
streamlit run app.py
```

Muốn dựng lại cả corpus từ nguồn gốc:

```bash
python -m src.task1_collect_legal_docs    # tải PDF Công báo + ghi sources.json
python -m src.task2_crawl_news            # crawl baochinhphu.vn
python -m src.task3_convert_markdown      # PDF/JSON -> Markdown chuẩn hoá
python -m src.task4_chunking_indexing     # chunk + embed + index
```

Mỗi module chạy độc lập được để kiểm tra từng bước:

```bash
python -m src.task5_semantic_search "câu hỏi"
python -m src.task6_lexical_search "câu hỏi"
python -m src.task7_reranking "câu hỏi"       # in cả vị trí dense# và bm25#
python -m src.task9_retrieval_pipeline "câu hỏi"  # in cosine gốc vs threshold
python -m src.task10_generation "câu hỏi"
```

## Kiểm tra

```bash
pytest tests/test_contracts.py -q     # 15 passed — signature, schema, thứ tự,
                                      # uniqueness, RRF, fallback, citation context
pytest tests/test_acceptance.py -q    # 5 passed  — số lượng dữ liệu, golden
                                      # dataset, báo cáo không còn placeholder
pytest -q                             # 20 passed
```

## Demo

```bash
python -m group_project.evaluation.demo            # cả ba phần, có gọi LLM
python -m group_project.evaluation.demo --no-llm   # chỉ retrieval, không tốn quota
```

Một lệnh ra đủ ba phần: query đúng domain, query ngoài domain và A/B. Transcript
của lần chạy đã ghi: [group_project/evaluation/DEMO.md](group_project/evaluation/DEMO.md).

| | Query | best dense cosine | Kết quả |
| --- | --- | ---: | --- |
| Đúng domain | "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?" | 0.7476 | trên ngưỡng 0.60 → hybrid, answer tách đúng 200 giờ/năm và ngoại lệ 300 giờ/năm, 4/5 chunk được trích dẫn |
| Ngoài domain | "Công thức nấu phở bò Hà Nội cần những nguyên liệu gì?" | 0.3523 | dưới ngưỡng → thử fallback → từ chối, **không nhãn `[Document N]` nào** trong answer |

Phần chat dùng `streamlit run app.py`: UI hiển thị answer, nguồn đã dùng, retrieval
method, score và chunk ID, đánh dấu chunk nào thật sự được trích dẫn, và cảnh báo
khi model bịa nhãn không có trong `sources`.

## Evaluation

Golden dataset 20 case (yêu cầu tối thiểu 15), A/B giữa **Config A — dense-only**
và **Config B — hybrid + RRF** trên cùng dataset, generator, prompt và `top_k`:

| Metric | Config A | Config B | Delta B−A |
| ------ | -------: | -------: | --------: |
| Faithfulness | 0.9536 | 0.9354 | −0.0182 |
| Answer relevance | 0.9378 | 0.9460 | +0.0082 |
| Context recall | 1.0000 | 0.9000 | −0.1000 |
| Context precision | 0.7581 | 0.8280 | +0.0699 |
| **Average** | **0.9124** | **0.9023** | **−0.0101** |

B xếp bằng chứng lên cao hơn (evidence rank trung bình 1.95 → 1.58, precision
+0.07) nhưng đánh mất bằng chứng ở 1/20 case mà A lấy được. Nguyên nhân đã truy
đến từng con số và ba worst performer đều có root cause riêng.

Báo cáo đầy đủ: [group_project/evaluation/RESULT.md](group_project/evaluation/RESULT.md).

```bash
python -m group_project.evaluation.run_evaluation                  # đủ, có gọi LLM
python -m group_project.evaluation.run_evaluation --retrieval-only # ~30 giây
```

## Cấu trúc thư mục

```
K4-L3B-RAG-Pipeline/
├── TEAMMATES.md                     thành viên, vai trò, link báo cáo cá nhân
├── README.md
├── app.py                           chatbot Streamlit
├── data/
│   ├── landing/                     PDF gốc + JSON bài báo + sources.json
│   └── standardized/                Markdown đã chuẩn hoá (10 file)
├── src/                             task1..task10 + contracts.py
├── tests/                           test_contracts.py, test_acceptance.py
├── group_project/evaluation/        golden_dataset, runner, demo, RESULT.md
├── reports/                         báo cáo cá nhân của từng thành viên
└── docs/                            contracts, step-by-step, rubric
```

Không commit: `.env`, `chroma_db/`, `data/pageindex/`, cache (`__pycache__/`,
`.pytest_cache/`, `*.egg-info/`) — xem [.gitignore](.gitignore).

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md) — schema, interface và invariant.
- [Step-by-step guide](docs/STEP_BY_STEP.md) — thứ tự triển khai từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md) — thang điểm.
- [Thành viên nhóm](TEAMMATES.md) — phân công và link tới báo cáo cá nhân
  (`reports/K4-L3B-<MSSV>-<Tên>.md`) của từng người.
- [Suggested topics](docs/SUGGESTED_TOPICS.md) — danh sách chủ đề tham khảo.
