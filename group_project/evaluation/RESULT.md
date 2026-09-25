# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | `ragas 0.4.3`, ChromaDB 1.5.9 |
| Evaluator model                    | `gemini-3.5-flash-lite` (Google AI Studio, endpoint tương thích OpenAI), `temperature=0` |
| Generator model                    | `openai/gpt-oss-120b` trên Groq, `temperature=0.3`, `top_p=0.9`, `max_tokens=2048` |
| Embedding model                    | `BAAI/bge-m3` qua `sentence_transformers`, 1024 chiều |
| Corpus version/commit              | code: cây làm việc tại thời điểm đo (`HEAD` = `a23df34`, working tree dirty), sau đó được commit nguyên trạng thành bản nộp bài; corpus `data/standardized/` gồm 10 file `.md`, SHA-256 `bcaff5eb818d6f65`, index 1180 chunk |
| Golden dataset size                | 20 case (yêu cầu tối thiểu 15) |
| `top_k`                            | 5 (giống nhau ở cả hai config) |
| Chunking                           | `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`, cắt theo heading rồi recursive-split |
| Fallback threshold and calibration | `SCORE_THRESHOLD=0.60` trên dense cosine similarity — xem mục "Hiệu chỉnh fallback threshold" |
| PageIndex fallback                 | Tắt ở cả hai config (`PAGEINDEX_API_KEY` trống), nên nhánh fallback không phải là biến gây nhiễu |
| Lệnh tái lập                       | `python -m group_project.evaluation.run_evaluation` |

`data/landing/` và `data/standardized/` được commit cùng repo, nên corpus dựng lại được từ chính commit nộp bài. Hai thứ **không** nằm trong git vì tái tạo được: `chroma_db/` (dựng lại bằng `python -m src.task4_chunking_indexing`) và `data/pageindex/` (PDF render lại được, `doc_id` gắn với tài khoản PageIndex của từng người).

SHA-256 ở trên vẫn được ghi vì commit hash một mình không bắt được trường hợp corpus bị sửa sau lần đo mà chưa chạy lại evaluation. Digest băm nội dung toàn bộ file Markdown đã chuẩn hoá; đổi một ký tự là digest đổi theo. Muốn so kết quả giữa hai lần chạy thì phải khớp cả digest này, không chỉ khớp commit.

## Configurations

- **Config A — dense-only:** `retrieve(query, top_k=5, use_reranking=False)`. Chỉ chạy `semantic_search` (ChromaDB cosine trên bge-m3), lấy 10 ứng viên rồi cắt còn 5. Không chạy BM25, không fuse.
- **Config B — hybrid + RRF:** `retrieve(query, top_k=5, use_reranking=True)`. Chạy `semantic_search` và `lexical_search` mỗi bên 10 ứng viên, fuse bằng `rerank_rrf([dense, sparse], top_k=5, k=60)`.

Mọi thứ còn lại giữ nguyên: cùng golden dataset, cùng generator và tham số sinh, cùng `SYSTEM_PROMPT`, cùng judge, cùng `top_k=5`, cùng `SCORE_THRESHOLD=0.60`, cùng nhánh fallback (tắt). Runner import thẳng `SYSTEM_PROMPT`, `format_context`, `reorder_for_llm`, `_citation_order` và `call_llm` từ `src/task10_generation.py` thay vì chép lại, nên prompt đưa vào LLM ở hai arm là cùng một đoạn code. Biến duy nhất đổi là cờ `use_reranking`.

## Hiệu chỉnh fallback threshold

`SCORE_THRESHOLD = 0.60`, so với **dense cosine similarity gốc** (`semantic_search`), không phải RRF score.

Con số này chỉ đúng cho corpus lao động hiện tại với embedding `BAAI/bge-m3`. Đổi corpus hoặc đổi embedding model thì phải đo lại: thang cosine của mỗi model một khác, và độ "gần" của corpus với câu hỏi ngoài domain cũng khác.

### Hai query hiệu chỉnh

| Loại | Query | Best dense cosine | Dưới ngưỡng? | Đường đi |
| ---- | ----- | ----------------: | ------------ | -------- |
| Trong domain | "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?" | 0.7476 | Không | hybrid; top-1 là Nghị quyết 17/2022 Điều 2 — đúng văn bản trả lời câu hỏi |
| Ngoài domain | "Công thức nấu phở bò Hà Nội cần những nguyên liệu gì?" | 0.3523 | Có | thử PageIndex fallback, fallback không bật nên giữ hybrid result |

### Khoảng trống đo được

Mỗi nhóm 5 query, lấy best dense cosine:

| Nhóm query | Khoảng | Ví dụ biên |
| ---------- | ------ | ---------- |
| Trong domain (5 query) | 0.7329 – 0.8022 | thấp nhất: "Nghị định 12/2022/NĐ-CP xử phạt vi phạm về làm thêm giờ ra sao?" (0.7329) |
| Ngoài domain (5 query) | 0.3523 – 0.4762 | cao nhất: "Học phí đại học ngành công nghệ thông tin năm 2024 là bao nhiêu?" (0.4762) |

Khoảng trống giữa hai nhóm là **0.4762 – 0.7329**. Chọn 0.60 ≈ điểm giữa, cách mỗi biên ~0.13 nên một query lệch nhẹ vẫn không nhảy sai nhánh. Query "học phí đại học" là ca sát biên nhất: nó vẫn là ngôn ngữ chính sách nên cosine cao hơn hẳn các query ngoài domain khác, đây là ứng viên gây false negative nếu sau này hạ ngưỡng xuống dưới 0.50.

### Vì sao không dùng RRF score cho quyết định này

Quan sát trực tiếp trong lần đo trên:

| Query | Best dense cosine | RRF score của top-1 |
| ----- | ----------------: | ------------------: |
| "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?" (trong domain) | 0.7476 | 0.032522 |
| "Tỷ số trận chung kết World Cup 2022 là bao nhiêu?" (ngoài domain) | 0.3923 | 0.032522 |

Hai query cách nhau 0.35 điểm cosine nhưng **RRF score bằng nhau đến từng chữ số**. RRF chỉ đọc vị trí trong danh sách: chunk nào đứng đầu cả dense lẫn BM25 cũng nhận `1/61 + 1/62 ≈ 0.0325`, bất kể nó có liên quan tới câu hỏi hay không. Dùng RRF score làm ngưỡng thì hoặc mọi query đều rơi vào fallback, hoặc không query nào rơi — không có ngưỡng nào phân biệt được hai dòng trên.

Vì vậy hai nhiệm vụ dùng hai score khác nhau: RRF hợp nhất *thứ hạng*, còn nhánh quyết định fallback đọc *cosine score gốc* của dense. Đặc tính "chỉ đọc thứ hạng" này quay lại đúng như một lỗi ở mục Worst performers bên dưới.

## Overall scores

Cả bốn metric đều chấm đủ 20/20 case ở cả hai config, nên delta dưới đây là phép so ghép cặp trên cùng một tập case.

| Metric            | Config A | Config B | Delta B−A | Case tốt lên / kém đi |
| ----------------- | -------: | -------: | --------: | --------------------- |
| Faithfulness      |   0.9536 |   0.9354 |   −0.0182 | 3 / 2 |
| Answer relevance  |   0.9378 |   0.9460 |   +0.0081 | 8 / 6 |
| Context recall    |   1.0000 |   0.9000 |   −0.1000 | 0 / 2 |
| Context precision |   0.7581 |   0.8280 |   +0.0699 | 11 / 4 |
| **Average**       |   0.9124 |   0.9023 |   −0.0101 | — |

Cột cuối cần đọc cùng cột delta. Answer relevance delta gần như bằng 0 (+0.008) nhưng có 14/20 case đổi điểm theo hai chiều gần cân nhau — đó là nhiễu, không phải "hai config cho kết quả giống nhau". Ngược lại context recall chỉ 2 case đổi, nhưng cả hai đều đổi theo chiều xấu và đủ nặng để kéo trung bình xuống 0.10.

### Chỉ số retrieval tất định

Bốn metric trên đều do LLM chấm. Bảng dưới không gọi LLM: `evidence_rank` là vị trí chunk đầu tiên chứa `expected_context` của golden case, so khớp chuỗi sau khi chuẩn hoá heading/emphasis/khoảng trắng. Chạy lại cho kết quả y hệt, nên đây là chỗ neo khi điểm LLM đáng ngờ.

| Chỉ số | Config A | Config B | Delta B−A |
| ------ | -------: | -------: | --------: |
| Evidence hit rate | 1.0000 (20/20) | 0.9500 (19/20) | −0.0500 |
| Evidence ở rank 1 | 12 | 14 | +2 |
| Evidence rank trung bình (chỉ case hit) | 1.95 | 1.58 | −0.37 |
| Chunk `legal/` trong top-5 (tổng 100 ô) | 67 | 62 | −5 |

Hai bảng nói cùng một chuyện từ hai phía: B xếp bằng chứng lên cao hơn (rank trung bình 1.95 → 1.58, số case bằng chứng ở rank 1 tăng 12 → 14), nhưng B đánh mất hoàn toàn bằng chứng ở 1 case mà A lấy được, và kéo thêm 5 ô nhiễu từ nguồn báo chí vào top-5.

### Latency và cost

| Chỉ số | Config A | Config B | Delta |
| ------ | -------: | -------: | ----: |
| Retrieval latency trung bình | 0.255 s | 0.300 s | +0.045 s (+17.5%) |
| Context đưa vào LLM (ký tự) | 4 746.8 | 4 786.4 | +39.6 (+0.8%) |
| Generation latency trung bình | 13.08 s | 15.05 s | +1.97 s |

Phần +0.045 s của B là BM25 cộng bước fuse, chạy hoàn toàn local nên **không phát sinh chi phí API**. Context gửi lên LLM chênh 0.8%, tức chi phí token mỗi câu trả lời gần như không đổi — cả hai config đều gửi đúng 5 chunk.

Chênh lệch generation latency **không quy được cho config**: hai arm gửi lượng context gần bằng nhau, khác biệt đến từ độ dài câu trả lời và độ trễ phía Groq giữa các lần gọi. Không nên dùng con số này để chọn config.

## A/B comparison

- **Cấu hình tốt hơn: Config B (hybrid + RRF), nhưng chỉ khi kèm bản vá cho lỗi mất bằng chứng ở mục Worst performers.** Ở trạng thái hiện tại, average của B thấp hơn A (0.9023 so với 0.9124) và không nên chọn B chỉ dựa trên average.

- **Evidence — B thắng ở chất lượng xếp hạng, thua ở độ phủ:** B đẩy context precision lên rõ rệt (+0.0699, 11 case tốt lên so với 4 case kém đi) và kéo bằng chứng lên cao hơn trong danh sách (rank trung bình 1.95 → 1.58). Đây đúng là việc RRF sinh ra để làm. Nhưng B làm context recall tụt 0.10 và đánh mất bằng chứng ở q18 — thứ A lấy được. Recall là metric mà lỗi không cứu được ở bước sau: generator không thể trích dẫn đoạn văn không có trong context.

- **Cùng hướng với chỉ số tất định:** context precision +0.0699 khớp với evidence rank trung bình giảm 0.37; context recall −0.10 khớp với evidence hit rate giảm 0.05. Hai hệ đo độc lập (LLM judge và so khớp chuỗi) cho cùng kết luận, nên kết luận này không phụ thuộc vào việc judge có chấm chuẩn hay không.

- **Trade-off latency/cost:** B đắt hơn 0.045 s mỗi truy vấn (+17.5% latency retrieval) và không tốn thêm tiền API. Với một chatbot mà tổng thời gian phản hồi bị chi phối bởi generation (~13–15 s), 0.045 s là không đáng kể. **Chi phí không phải là lý do để loại B.**

- **Lỗi còn tồn tại ở cả hai config:** context precision là metric thấp nhất ở cả A (0.7581) và B (0.8280) — top-5 luôn chứa nhiễu. Ba case tệ nhất về precision đều dưới 0.5 ở ít nhất một arm (q16, q17, q18). Nâng B lên không xoá được vấn đề này.

### Phân tích theo nhóm câu hỏi

Golden dataset gắn nhãn `retrieval_challenge` theo chủ đích thiết kế câu hỏi, gán trước khi chạy pipeline:

| Nhóm | n | Avg A | Avg B | Delta | Evidence rank A → B |
| ---- | -: | ----: | ----: | ----: | ------------------- |
| `lexical_keyword` (câu hỏi dùng đúng thuật ngữ/số hiệu văn bản) | 8 | 0.915 | 0.925 | **+0.010** | 1.75 → **1.12** |
| `source_confusion` (chủ đề có cả trong báo lẫn trong luật) | 7 | 0.914 | 0.908 | −0.006 | 2.00 → **1.50** |
| `semantic_paraphrase` (câu hỏi diễn đạt lại, không lặp thuật ngữ) | 5 | 0.907 | 0.858 | **−0.049** | 2.20 → 2.40 |

Đây là bảng giải thích toàn bộ phần còn lại của báo cáo. **RRF giúp đúng ở nhóm BM25 đóng góp được, và hại đúng ở nhóm BM25 mù.** Nhóm `lexical_keyword` là nhóm duy nhất B thắng cả điểm lẫn thứ hạng. Nhóm `semantic_paraphrase` là nhóm duy nhất B thua, và cũng là nhóm chứa cả hai case mà BM25 không tìm ra bằng chứng trong top-5 (q02 và q18).

Đo trực tiếp đóng góp của từng nhánh trên 20 case: BM25 tìm được bằng chứng trong top-5 ở 18/20 case, dense ở 20/20. Hai case BM25 trượt là **đúng hai case Config B tụt hạng nặng nhất**.

## Worst performers

Ba case chọn theo điểm trung bình 4 metric thấp nhất ở Config B.

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | q18 — "Những khoảng thời gian nào được tính vào thời giờ làm việc được hưởng lương?" | B (A: 0.797) | 1.00 | 0.99 | **0.00** | **0.00** | **retrieval** | RRF chỉ đọc thứ hạng nên chunk được cả hai nhánh cùng xếp hạng luôn ăn điểm kép, đẩy bằng chứng chỉ-dense ra khỏi top-5 |
|   2 | q02 — "Nếu doanh nghiệp quy định thời giờ làm việc theo tuần thì mỗi ngày được làm tối đa bao nhiêu giờ?" | B (A: 0.924) | **0.00** | 0.92 | 1.00 | 0.70 | **evaluation** (judge false negative), kèm regression retrieval nhẹ | Câu trả lời đúng và có trong context, judge chấm sai; riêng phần retrieval bằng chứng tụt rank 4 → 5 |
|   3 | q05 — "Những ngành nghề nào được làm thêm tới 300 giờ trong 01 năm?" | B (A: 0.866) | 1.00 | 0.94 | **0.00** | 1.00 | **data** (chunking), bị retrieval khuếch đại | Danh sách ngành nghề của Điều 107 khoản 3 bị cắt qua hai chunk; B bỏ mất chunk dự phòng che được phần thiếu |

### q18 — mất bằng chứng do cơ chế cộng điểm của RRF

Bằng chứng cần lấy là Nghị định 145/2020 **Điều 58**. Config A lấy được ở rank 4. Config B trượt hoàn toàn: cả 5 kết quả đều là chunk của `news/article_04.md`.

Nguyên nhân có thể truy đến từng con số. Tiêu đề bài báo `article_04` ("Thời gian nghỉ trong giờ làm việc có được tính trả lương?") trùng gần như nguyên văn câu hỏi, và chunker gắn tiêu đề đó vào mọi chunk của bài. Hệ quả là BM25 top-10 có **4 chunk của cùng một bài báo, và không hề có Điều 58**:

```
BM25 top-10 (q18)
 #1  29.447  news/article_04.md::chunk-8
 #2  25.780  news/article_04.md::chunk-6
 #3  25.037  news/article_01.md::chunk-2
 ...
 #7  22.500  news/article_04.md::chunk-7
 #8  21.825  news/article_04.md::chunk-5
```

RRF cho mỗi chunk `1/(60 + rank)` trên từng danh sách, rồi cộng lại. Chunk xuất hiện ở **cả hai** danh sách nhận điểm kép; chunk chỉ có ở một danh sách thì không:

| Chunk | Có mặt ở | RRF score |
| ----- | -------- | --------: |
| `news/article_04` × 3 | dense + BM25 | 0.031545 / 0.031054 / 0.030835 |
| `news/article_04::chunk-1` | 1 danh sách, rank 1 | 1/61 = 0.016393 |
| `news/article_01::chunk-2` | 1 danh sách, rank 3 | 1/63 = 0.015873 |
| **`nghi_dinh_145::chunk-192` (bằng chứng)** | **chỉ dense, rank 4** | **1/64 = 0.015625** |

Bằng chứng rơi xuống hạng 6 — lệch top-5 đúng một bậc, thua chunk đứng trên nó 0.000248 điểm. Không phải dense xếp sai: dense vẫn để Điều 58 ở rank 4. Lỗi nằm ở chỗ RRF cho một nhánh mù hoàn toàn về câu hỏi này quyền phủ quyết ngang với nhánh tìm đúng.

Hệ quả xuống tầng generation quan sát được trực tiếp: câu trả lời của B trích 30 phút / 45 phút nghỉ giữa giờ lấy từ bài báo, tức trả lời một câu hỏi lân cận chứ không phải câu được hỏi. Faithfulness vẫn 1.00 vì answer đúng là bám context — đây là ví dụ rõ cho việc faithfulness cao **không** đảm bảo câu trả lời đúng khi context sai.

### q02 — lỗi của phép đo, không phải lỗi của hệ thống

Faithfulness 0.00, nhưng đọc lại thì câu trả lời đúng và bằng chứng có trong context:

- Answer: "Nếu doanh nghiệp quy định thời giờ làm việc theo tuần, thời giờ làm việc bình thường không được vượt quá **10 giờ trong một ngày**. [Document 5] [Document 2]"
- Document 5 = `bo_luat_lao_dong_2019.md::chunk-163`, chứa nguyên văn: "trường hợp theo tuần thì thời giờ làm việc bình thường không quá 10 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần."

Cùng khẳng định đó ở Config A được chấm faithfulness **1.00**. Judge `gemini-3.5-flash-lite` chấm không nhất quán giữa hai arm trên cùng một nội dung. Một mình case này kéo delta faithfulness của nhóm `semantic_paraphrase` xuống −0.233; bỏ nó ra thì faithfulness B không còn thua A.

Phần thuộc về hệ thống trong case này nhỏ hơn nhiều: bằng chứng tụt rank 4 → 5 và context precision 0.81 → 0.70, cùng cơ chế với q18 nhưng chưa đủ mạnh để đẩy hẳn ra ngoài top-5.

Đây là lý do báo cáo giữ cả `evidence_rank` tất định lẫn điểm LLM: nếu chỉ nhìn bảng metric, nhóm sẽ đi sửa prompt generation cho một lỗi mà generation không gây ra.

### q05 — chunk cắt đứt danh sách điều khoản

Context recall 0.00 dù bằng chứng nằm ở **rank 1**. Hai con số không mâu thuẫn, chúng đo hai thứ khác nhau, và chỗ lệch chính là lỗi.

Chunk `bo_luat_lao_dong_2019.md::chunk-166` chứa phần đầu khoản 3 Điều 107 (điểm a, b, c) nhưng **thiếu điểm d và đ** — chunk dài 765 ký tự, phần còn lại của khoản rơi sang chunk kế tiếp. `evidence_rank` khớp được vì nó neo theo 200 ký tự đầu của `expected_context`; judge chấm recall theo toàn bộ danh sách trong `expected_answer` nên thấy thiếu.

Config A vẫn đạt recall 1.00 vì nó lấy thêm `nghi_dinh_145_2020.md::chunk-197` (Điều 61 — "Các trường hợp được tổ chức làm thêm từ trên 200 giờ đến 300 giờ") ở rank 1, văn bản này liệt kê lại cùng nhóm ngành nghề và che được phần thiếu. Config B thay chunk đó bằng `nghi_quyet_17_2022.md::chunk-3` (Điều 2 — trần giờ theo tháng), không liên quan tới câu hỏi.

Lỗi gốc là **data/chunking**: cắt cứng theo `CHUNK_SIZE` làm đứt một danh sách điều khoản phải đọc trọn mới trả lời đúng. Retrieval chỉ khuếch đại nó — A tình cờ vá được nhờ một nguồn trùng lặp, B thì không. Nếu chunk giữ trọn khoản 3, cả hai config đều không gặp lỗi này.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
| 1 | Giới hạn số chunk mỗi `doc_id` trong danh sách đưa vào `rerank_rrf` (source diversity cap, ví dụ tối đa 2 chunk/tài liệu) | q18: BM25 top-10 có 4/10 chunk cùng thuộc `news/article_04.md`, đẩy Điều 58 khỏi top-5 | q18 lấy lại bằng chứng → evidence hit rate B về 20/20, context recall B từ 0.90 lên ~0.95; giảm 5 ô nhiễu news trong top-5 | `python -m group_project.evaluation.run_evaluation --retrieval-only` rồi so `evidence_rank` của q18 và cột "chunk `legal/` trong top-5" với bảng "Chỉ số retrieval tất định" ở trên. Không cần gọi LLM. |
| 2 | Cho RRF trọng số theo độ tin cậy từng nhánh, hoặc bỏ đóng góp của BM25 khi best BM25 score của query thấp hơn ngưỡng | Nhóm `semantic_paraphrase` là nhóm duy nhất B thua A (−0.049); đúng 2 case BM25 không tìm ra bằng chứng (q02, q18) là 2 case B tụt hạng nặng nhất | Giữ nguyên phần B đang thắng (`lexical_keyword` +0.010, precision +0.0699) mà không mất recall ở nhóm paraphrase | Chạy lại A/B đầy đủ, so bảng "Phân tích theo nhóm câu hỏi": delta nhóm `semantic_paraphrase` phải ≥ 0 trong khi delta `lexical_keyword` không giảm |
| 3 | Chunk theo đơn vị điều/khoản thay vì cắt cứng `CHUNK_SIZE=1000`, hoặc cho một khoản luôn nằm trọn trong một chunk | q05: `chunk-166` dài 765 ký tự nhưng thiếu điểm d, đ của khoản 3 Điều 107 → context recall B = 0.00 dù bằng chứng ở rank 1 | q05 hết phụ thuộc vào việc có vớ được nguồn trùng lặp hay không; các câu hỏi `multi_fact` khác (q04, q12, q17) cũng bớt rủi ro cùng kiểu | Sau khi reindex, kiểm chunk chứa Điều 107 khoản 3 có đủ điểm a–đ, rồi chạy lại và so `context_recall` của q05 ở cả hai config. **Lưu ý:** reindex đổi corpus SHA-256, nên số mới chỉ so được với một baseline chạy lại trên cùng index. |
| 4 | Chấm lại bằng judge mạnh hơn, hoặc chấm 2 lần rồi lấy trung bình | q02: faithfulness 0.00 ở B nhưng 1.00 ở A cho cùng một khẳng định có nguyên văn trong context; 19/160 ô bị lỗi parse ở lượt chấm đầu, phải chấm bù | Bỏ nhiễu judge khỏi delta faithfulness; hiện −0.0182 gần như do một mình q02 tạo ra | `EVAL_JUDGE_MODEL=<model khác> python -m group_project.evaluation.run_evaluation --score-only` rồi so `per_case.B.q02.metrics.faithfulness` trong `scores.json`. Dùng `--score-only` để chấm lại đúng bộ câu trả lời cũ, không sinh lại. |
| 5 | Đặt `TEMPERATURE = 0` trong `src/task10_generation.py` cho các lần chạy đo | Generator đang chạy `temperature=0.3`, nên chạy lại sinh câu trả lời khác và faithfulness/relevance lệch theo, không tách được khỏi tác động của config | Delta B−A tái lập được giữa các lần chạy | Chạy `run_evaluation` hai lần liên tiếp không đổi gì khác, so `summary` trong `scores.json`; hai lần phải trùng nhau |

Ưu tiên 1 và 2 cùng nhắm vào một lỗi (q18) từ hai hướng, và nên làm 1 trước: nó rẻ hơn, kiểm chứng được mà không tốn token LLM nào.

## Hạn chế của phép đo

Ghi lại để người đọc sau biết chỗ nào nên tin và chỗ nào nên kiểm lại:

- **Judge nhẹ.** `gemini-3.5-flash-lite` được chọn vì ràng buộc hạn mức, không vì chất lượng. Chấm 4 metric tốn 9 request và ~12 500 token mỗi case, tức ~500 000 token cho 20 case × 2 config; các phương án còn lại đều chặn: `qwen/qwen3.8-27b` giới hạn 1 000 output token/phút trong khi một request faithfulness cần ~1 600; `openai/gpt-oss-120b` có 200 000 token/ngày và đã bị 40 lời gọi generation dùng gần hết; `gemini-3.5-flash` free tier chỉ cho 20 request/ngày. Case q02 cho thấy judge này chấm sai được. **Điểm tuyệt đối nên đọc dè dặt; delta B−A vững hơn** vì hai arm dùng chung judge, chung generator, chung prompt và chung `top_k`.
- **Judge không ổn định giữa các lần chấm.** Lượt chấm đầu có 19/160 ô trả về JSON hỏng, lệch hẳn về Config B (17 ô) so với A (2 ô). Đã vá bằng `--fill-gaps` để cả bốn metric đủ 20/20 ở hai arm, nhưng các ô vá được chấm ở lượt gọi khác lượt đầu.
- **Generator không tất định.** `temperature=0.3` — xem recommendation 5.
- **n = 20.** Đủ để thấy một lỗi mất bằng chứng và một lỗi chunking, không đủ để coi delta cỡ 0.01 là có ý nghĩa. Cụ thể: average (−0.0101), faithfulness (−0.0182) và answer relevance (+0.0081) đều nằm trong khoảng nhiễu. Chỉ context recall (−0.10) và context precision (+0.07) là đủ lớn để kết luận, và cả hai đều được chỉ số tất định xác nhận độc lập.
- **Nhãn `retrieval_challenge` là chủ đích thiết kế, không phải đo đạc.** Nhãn được gán theo cách đặt câu hỏi trước khi chạy pipeline, nên nó là giả thuyết được dữ liệu ủng hộ, không phải kết quả rút ra từ dữ liệu.
- **Index không nằm trong git.** Corpus (`data/landing/`, `data/standardized/`) thì có, nhưng `chroma_db/` phải dựng lại trước khi chạy lại evaluation. Cùng corpus và cùng tham số chunking thì index dựng lại là tất định, nên SHA-256 trong `scores.json` vẫn là thứ ràng buộc kết quả với dữ liệu.

## Bonus experiments

Chưa thực hiện. Nhóm không làm HyDE/query expansion, reranker mô hình hoá, conversation memory hay deploy online trong phạm vi lần đo này, nên không có baseline, metric delta và số đo latency/cost để báo cáo. Hạ tầng đo đã sẵn: thêm một entry vào `CONFIGS` trong `group_project/evaluation/run_evaluation.py` là chạy được cùng một quy trình A/B trên cùng golden dataset.

## Tái lập

```bash
# 1. Index corpus (cần khi chroma_db/ chưa có hoặc corpus đã đổi)
python -m src.task4_chunking_indexing

# 2. Chạy đủ: retrieval + generation + chấm 4 metric cho cả hai config
python -m group_project.evaluation.run_evaluation

# Chỉ phần tất định, không gọi LLM, chạy trong ~30 giây
python -m group_project.evaluation.run_evaluation --retrieval-only

# Chấm lại đúng bộ câu trả lời đã sinh trong runs.json (không sinh lại)
python -m group_project.evaluation.run_evaluation --score-only

# Chấm bù các ô NaN còn lại, giữ nguyên ô đã có điểm
python -m group_project.evaluation.run_evaluation --fill-gaps

# Demo: query đúng domain, query ngoài domain và A/B (transcript trong DEMO.md)
python -m group_project.evaluation.demo
python -m group_project.evaluation.demo --no-llm   # chỉ retrieval, không tốn quota
```

Cấu hình judge đọc từ `.env` (`EVAL_JUDGE_MODEL`, `EVAL_JUDGE_BASE_URL`, `EVAL_JUDGE_KEY_NAME`, `EVAL_MAX_WORKERS`) nên đổi evaluator không phải sửa code. Xem `.env.example`.

File sinh ra:

- `runs.json` — retrieval và câu trả lời từng case của cả hai config, kèm metadata run.
- `scores.json` — điểm từng case, tổng hợp, và `paired_delta` (delta chỉ tính trên case cả hai arm đều chấm được).

Transcript demo (query đúng domain, query ngoài domain, A/B trên chính query demo)
nằm trong [DEMO.md](DEMO.md).
