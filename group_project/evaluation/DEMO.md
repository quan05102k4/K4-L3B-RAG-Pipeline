# Demo — query đúng domain, query ngoài domain và A/B

Bản ghi này là transcript của một lần chạy thật, để người chấm đối chiếu trước
khi xem demo trực tiếp. Chạy lại bằng một lệnh:

```bash
python -m group_project.evaluation.demo            # cả ba phần, có gọi LLM
python -m group_project.evaluation.demo --no-llm   # chỉ retrieval, không tốn quota
python -m group_project.evaluation.demo --part ab  # chỉ phần A/B
```

Điều kiện: `chroma_db/` đã được dựng (`python -m src.task4_chunking_indexing`) và
`.env` có `GROQ_API_KEY`. Phần `--no-llm` chỉ cần ChromaDB.

Cấu hình của lần chạy dưới đây:

```
Generator : groq / openai/gpt-oss-120b
Embedding : sentence_transformers / BAAI/bge-m3
Chunking  : size=1000 overlap=150
Retrieval : top_k=5 score_threshold=0.6
```

Trong buổi demo, phần chat dùng `streamlit run app.py` — UI hiển thị cùng bộ
`sources` mà script in ra, kèm đánh dấu chunk nào thật sự được trích dẫn.

---

## 1. Query đúng domain

**Query:** *Làm thêm giờ tối đa bao nhiêu giờ trong một năm?*

```
best dense cosine = 0.7476   SCORE_THRESHOLD = 0.6
-> TRÊN ngưỡng => giữ hybrid, không gọi fallback

retrieval_source = hybrid | sources = 5 | tu choi = khong
```

Answer (rút gọn phần định dạng):

```
Theo quy định của pháp luật lao động Việt Nam, thời gian làm thêm tối đa trong
một năm như sau:

1. Mức chuẩn: Người lao động không được làm thêm quá 200 giờ trong 01 năm.
   - [Document 3] "Bảo đảm số giờ làm thêm của người lao động không quá 200 giờ
     trong 01 năm, trừ trường hợp quy định tại khoản 3 Điều này."
   - [Document 4] "không quá 200 giờ trong năm."

2. Trường hợp đặc biệt: ... thời gian làm thêm có thể lên tới 300 giờ trong 01 năm.
   - [Document 3] "Người sử dụng lao động được sử dụng người lao động làm thêm
     không quá 300 giờ trong 01 năm trong một số ngành, nghề, công việc..."
   - [Document 1] "Trường hợp người sử dụng lao động được sử dụng người lao động
     làm thêm tối đa 300 giờ trong 01 năm ... không quá 60 giờ trong 01 tháng."
   - [Document 5] "trừ một số trường hợp đặc biệt do Chính phủ quy định thì được
     làm thêm giờ không quá 300 giờ trong 1 năm."

Vì vậy, giới hạn chung là 200 giờ/năm, nhưng có thể lên tới 300 giờ/năm trong
các trường hợp được pháp luật cho phép.
```

Nguồn đã đưa vào context — nhãn `[Document N]` trỏ đúng `sources[N-1]`:

| Nhãn | Được trích | Score (RRF) | Tài liệu | Chunk ID |
| ---- | ---------- | ----------: | -------- | -------- |
| [Document 1] | có | 0.032522 | Nghị quyết 17/2022/UBTVQH15 | `legal/nghi_quyet_17_2022.md::chunk-3` |
| [Document 2] | không | 0.030777 | Nghị định 145/2020/NĐ-CP | `legal/nghi_dinh_145_2020.md::chunk-196` |
| [Document 3] | có | 0.030366 | Bộ luật Lao động 45/2019/QH14 | `legal/bo_luat_lao_dong_2019.md::chunk-166` |
| [Document 4] | có | 0.029911 | Báo Chính phủ — "Làm thêm giờ thế nào là đúng quy định?" | `news/article_02.md::chunk-3` |
| [Document 5] | có | 0.029857 | Báo Chính phủ — "Cách tính tiền lương làm ca đêm và thêm giờ" | `news/article_03.md::chunk-3` |

Ba điểm để đối chiếu khi demo:

1. **Citation truy được về nguồn gốc.** `[Document 3]` → `bo_luat_lao_dong_2019.md::chunk-166`
   → `data/standardized/legal/bo_luat_lao_dong_2019.md` (Điều 107 khoản 2–3)
   → PDF Công báo trong `data/landing/legal/` với `sha256` ghi trong `sources.json`.
2. **Chunk không được trích vẫn hiển thị.** `[Document 2]` nằm trong context nhưng
   model không dùng; UI đánh dấu "không được trích dẫn" thay vì giấu đi, nên người
   đọc thấy được hệ thống đưa vào những gì chứ không chỉ thấy phần thuận lợi.
3. **Answer phân biệt được mức chung 200 giờ và ngoại lệ 300 giờ**, mỗi mức dẫn
   nhãn riêng — đây là câu hỏi mà một câu trả lời gộp sẽ sai.

## 2. Query ngoài domain

**Query:** *Công thức nấu phở bò Hà Nội cần những nguyên liệu gì?*

```
best dense cosine = 0.3523   SCORE_THRESHOLD = 0.6
-> DƯỚI ngưỡng => thử PageIndex fallback
   PAGEINDEX_API_KEY trống => fallback trả rỗng => giữ hybrid result

retrieval_source = hybrid | sources = 5 | tu choi = CO (model tu choi, khong trich nhan nao)

--- Answer ---
Không thể xác minh câu trả lời vì không có thông tin về công thức nấu phở bò Hà
Nội trong các nguồn hiện có.
```

Cả 5 chunk trả về đều mang nhãn "không được trích dẫn": Bộ luật Lao động chunk-148,
Nghị định 145/2020 chunk-424 / chunk-150 / chunk-269, và một bài báo về thời gian
nghỉ giữa giờ. Không nhãn `[Document N]` nào xuất hiện trong answer.

**Điểm cần nói rõ khi demo — hệ thống có hai đường từ chối khác nhau:**

| Đường | Kích hoạt khi | Phụ thuộc model? |
| ----- | ------------- | ---------------- |
| Safe refusal tất định | `retrieve()` trả rỗng, hoặc provider lỗi | Không — pipeline trả `REFUSAL_ANSWER` cố định |
| Model từ chối | Có chunk nhưng không chunk nào trả lời được câu hỏi | Có — `SYSTEM_PROMPT` là thứ chặn |

Query "phở bò" đi **đường thứ hai**. Dense score 0.3523 dưới ngưỡng nên pipeline
thử fallback, nhưng `retrieve()` vẫn trả về 5 chunk — hybrid luôn xếp hạng được
một top-5 kể cả khi không chunk nào liên quan, vì RRF chỉ đọc thứ hạng. Vì vậy
nhánh safe refusal tất định không kích hoạt, và thứ chặn câu trả lời bịa là
`SYSTEM_PROMPT`. Tín hiệu kiểm được bằng máy là **answer không trích nhãn nào**:
không có `[Document N]` nghĩa là không khẳng định nào được neo vào context.

Đây là một hạn chế thật của thiết kế hiện tại, không phải chi tiết trình bày:
nếu đổi sang một model yếu hơn về instruction-following, query ngoài domain có
thể nhận được câu trả lời bịa mà ngưỡng 0.60 không chặn được, vì ngưỡng hiện chỉ
điều khiển việc *thử fallback* chứ không cắt kết quả. Hướng vá: khi
`best_dense_score` dưới ngưỡng và fallback không trả về gì, trả thẳng safe refusal
thay vì giữ hybrid result.

## 3. A/B — Config A (dense-only) vs Config B (hybrid + RRF)

### Trên chính query demo

Cùng query, cùng `top_k=5`, chỉ đổi cờ `use_reranking`. 3/5 chunk trùng nhau:

```
Config A — dense-only (use_reranking=False)
  A1. [dense 0.747584] legal/nghi_quyet_17_2022.md::chunk-2
  A2. [dense 0.721056] legal/nghi_quyet_17_2022.md::chunk-3
  A3. [dense 0.713991] legal/bo_luat_lao_dong_2019.md::chunk-166
  A4. [dense 0.707705] news/article_02.md::chunk-3
  A5. [dense 0.700666] legal/nghi_dinh_145_2020.md::chunk-197

Config B — hybrid + RRF (use_reranking=True)
  B1. [hybrid 0.032522] legal/nghi_quyet_17_2022.md::chunk-3      <- A#2
  B2. [hybrid 0.030777] legal/nghi_dinh_145_2020.md::chunk-196    <- mới (BM25 kéo lên)
  B3. [hybrid 0.030366] legal/bo_luat_lao_dong_2019.md::chunk-166 <- A#3
  B4. [hybrid 0.029911] news/article_02.md::chunk-3               <- A#4
  B5. [hybrid 0.029857] news/article_03.md::chunk-3               <- mới (BM25 kéo lên)

Bị B loại khỏi top-5: nghi_quyet_17_2022.md::chunk-2, nghi_dinh_145_2020.md::chunk-197
```

Hai thang score không so sánh trực tiếp được với nhau: A hiển thị cosine
similarity (0.70–0.75), B hiển thị RRF score (~0.03). Đây đúng là lý do RRF chỉ
dùng để gộp *thứ hạng*, còn quyết định fallback đọc cosine gốc của dense.

### Trên toàn golden dataset (20 case, đã chấm)

| Metric | Config A | Config B | Delta B−A |
| ------ | -------: | -------: | --------: |
| Faithfulness | 0.9536 | 0.9354 | −0.0182 |
| Answer relevance | 0.9378 | 0.9460 | +0.0082 |
| Context recall | 1.0000 | 0.9000 | −0.1000 |
| Context precision | 0.7581 | 0.8280 | +0.0699 |
| **Average** | **0.9124** | **0.9023** | **−0.0101** |
| Evidence hit rate (tất định) | 1.0000 | 0.9500 | −0.0500 |
| Evidence rank TB (tất định) | 1.95 | 1.58 | −0.37 |
| Retrieval latency (s) | 0.2549 | 0.2996 | +0.0447 |

Kết luận ngắn để nói trong demo: **B xếp bằng chứng lên cao hơn (rank 1.95 → 1.58,
precision +0.07) nhưng đánh mất bằng chứng ở 1/20 case mà A lấy được (recall
−0.10).** Nguyên nhân đã truy đến từng con số ở q18: RRF cho chunk xuất hiện ở cả
hai danh sách điểm kép, nên bằng chứng chỉ-dense thua 0.000248 điểm và rớt khỏi
top-5. Vì vậy nhóm khuyến nghị B **kèm** source-diversity cap, không phải B như
hiện tại.

Phân tích đầy đủ — worst performers, root cause từng case, 5 recommendations và
hạn chế của phép đo — nằm trong [RESULT.md](RESULT.md).
