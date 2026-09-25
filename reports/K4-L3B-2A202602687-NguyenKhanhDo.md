# Individual contribution report

## Thông tin

- Họ và tên: Nguyễn Khánh Đô
- Mã học viên: 2A202602687
- Nhóm: K4-L3B — chatbot RAG pháp luật lao động Việt Nam
- Repository/branch: https://github.com/quan05102k4/K4-L3B-RAG-Pipeline — `main`
- Vai trò: Thành viên — Evaluation và Integration

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Golden dataset | Soạn 20 case (yêu cầu tối thiểu 15) neo vào corpus thật, mỗi case có `expected_context` trích nguyên văn từ `data/standardized/`; gắn nhãn `retrieval_challenge` trước khi chạy pipeline: 8 `lexical_keyword`, 7 `source_confusion`, 5 `semantic_paraphrase` | `group_project/evaluation/golden_dataset.json` | Done |
| A/B evaluation harness | Viết runner chỉ đổi đúng một biến (`use_reranking`); import thẳng `SYSTEM_PROMPT`, `format_context`, `reorder_for_llm`, `_citation_order`, `call_llm` từ Task 10 thay vì chép lại, để prompt hai arm là cùng một đoạn code | `group_project/evaluation/run_evaluation.py` (652 dòng) | Done |
| Chỉ số retrieval tất định | Thêm `evidence_rank` / `evidence_hit`: vị trí chunk đầu tiên chứa `expected_context`, so khớp chuỗi sau khi chuẩn hoá heading/emphasis/khoảng trắng — không gọi LLM, chạy lại cho kết quả y hệt | `run_evaluation.py`, cột trong `scores.json` | Done |
| Chế độ chạy từng phần | `--retrieval-only` (không gọi LLM, ~30 giây), `--score-only` (chấm lại đúng bộ answer cũ), `--fill-gaps` (chấm bù ô NaN, giữ ô đã có điểm) | `run_evaluation.py` | Done |
| Task 8 — PageIndex fallback | Gọi REST trực tiếp thay vì `PageIndexClient`; cache `doc_id` theo sha256 nội dung; deadline chung cho cả vòng retrieval | `src/task8_pageindex_vectorless.py` | Done (fallback tắt ở lần đo — không có API key) |
| Báo cáo evaluation | Viết RESULT.md: run information, hiệu chỉnh threshold, overall scores, A/B, phân tích theo nhóm câu hỏi, 3 worst performers truy đến root cause, 5 recommendations, hạn chế của phép đo | `group_project/evaluation/RESULT.md` | Done |
| Demo runner | Script chạy một lệnh ra đủ ba phần demo (in-domain, out-of-domain, A/B) kèm transcript đã ghi | `group_project/evaluation/demo.py`, `group_project/evaluation/DEMO.md` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Tách judge khỏi generator — chấm bằng `gemini-3.5-flash-lite` qua endpoint tương thích OpenAI của Google, trong khi generator là `openai/gpt-oss-120b` trên Groq.
   **Lý do/evidence:** Hai lý do đo được, không phải theo lệ. (a) Self-preference bias: chấm bằng chính model đã sinh câu trả lời đẩy faithfulness và answer relevance cao hơn thực tế. (b) Ngân sách token: chấm 4 metric tốn ~12.500–14.750 token/case, tức ~500–590k cho 20 case × 2 config, trong khi Groq free tier cho 200k/ngày/model và 40 lời gọi generation đã ăn gần hết hạn mức. Dồn cả hai việc vào một model thì run chấm chết giữa chừng vì 429. Ba biến `EVAL_JUDGE_MODEL`, `EVAL_JUDGE_BASE_URL`, `EVAL_JUDGE_KEY_NAME` đọc từ `.env` nên đổi evaluator không phải sửa code.
   **Trade-off:** `flash-lite` được chọn vì hạn mức chứ không vì chất lượng, và nó chấm sai thật: q02 nhận faithfulness 0.00 ở Config B nhưng 1.00 ở Config A cho **cùng một khẳng định có nguyên văn trong context**. Vì vậy báo cáo nói rõ điểm tuyệt đối phải đọc dè dặt, còn delta B−A vững hơn vì hai arm dùng chung judge.

2. **Quyết định:** Đo thêm một tầng chỉ số **tất định** song song với 4 metric LLM, thay vì chỉ báo cáo điểm RAGAS.
   **Lý do/evidence:** Hai case cho thấy điểm LLM một mình sẽ dẫn nhóm đi sửa nhầm chỗ. q02: faithfulness 0.00 nhưng `evidence_rank` = 5 và answer trích đúng nguyên văn — lỗi của phép đo, không phải của hệ thống; nếu chỉ nhìn bảng metric, nhóm sẽ đi sửa prompt generation cho một lỗi mà generation không gây ra. q05: `context_recall` 0.00 trong khi `evidence_rank` = 1 — chỗ lệch giữa hai hệ đo chính là lỗi chunking. Ngược lại, khi hai hệ đo độc lập cùng chỉ một hướng (precision +0.0699 ↔ evidence rank 1.95 → 1.58; recall −0.10 ↔ hit rate −0.05) thì kết luận không còn phụ thuộc vào việc judge chấm chuẩn hay không.
   **Trade-off:** `evidence_rank` neo theo 200 ký tự đầu của `expected_context`, nên nó báo "hit" cả khi chunk chứa phần đầu mà thiếu phần đuôi — đúng ca q05. Chỉ số này bắt được *có lấy đúng tài liệu không*, không bắt được *có lấy đủ không*; phải đọc cùng `context_recall` chứ không thay thế được.

## Kiểm thử và kết quả

- **Ba lệnh kiểm tra:** `pytest tests/test_contracts.py -q` → 15 passed; `pytest tests/test_acceptance.py -q` → 5 passed; `pytest -q` → **20 passed**. `test_evaluation_report_is_completed` chặn placeholder: RESULT.md không còn chữ "TODO" nào và phải có đủ 4 heading bắt buộc.
- **Kết quả A/B (20 case, cùng golden dataset / generator / prompt / `top_k=5`):**

  | Metric | Config A (dense-only) | Config B (hybrid + RRF) | Delta B−A |
  | --- | ---: | ---: | ---: |
  | Faithfulness | 0.9536 | 0.9354 | −0.0182 |
  | Answer relevance | 0.9378 | 0.9460 | +0.0082 |
  | Context recall | 1.0000 | 0.9000 | −0.1000 |
  | Context precision | 0.7581 | 0.8280 | +0.0699 |
  | Average | 0.9124 | 0.9023 | −0.0101 |
  | Evidence hit rate (tất định) | 1.0000 | 0.9500 | −0.0500 |
  | Evidence rank TB (tất định) | 1.95 | 1.58 | −0.37 |

- **Phân tích theo nhóm câu hỏi** — đây là bảng giải thích toàn bộ phần còn lại: B thắng đúng ở nhóm BM25 đóng góp được (`lexical_keyword` +0.010, evidence rank 1.75 → 1.12) và thua đúng ở nhóm BM25 mù (`semantic_paraphrase` −0.049). Đo trực tiếp: BM25 tìm được bằng chứng trong top-5 ở 18/20 case, dense 20/20, và **đúng 2 case BM25 trượt (q02, q18) là 2 case Config B tụt hạng nặng nhất**.
- **Lỗi đã phát hiện và cách xử lý:**
  - Lượt chấm đầu có **19/160 ô trả về JSON hỏng**, lệch hẳn về Config B (17 ô) so với A (2 ô) — nếu bỏ qua thì delta B−A tính trên hai tập case khác nhau và vô nghĩa. Tôi viết `--fill-gaps` để chấm bù đúng các ô NaN mà giữ nguyên ô đã có điểm; kết quả cuối cả 4 metric đều đủ 20/20 ở hai arm, nên delta là phép so ghép cặp thật.
  - Generator chạy `temperature=0.3` nên chạy lại sinh câu trả lời khác và điểm lệch theo. Tôi ghi việc này thành recommendation 5 (đặt `TEMPERATURE = 0` cho các lần chạy đo) thay vì lặng lẽ báo cáo con số không tái lập được.
  - Corpus SHA-256 (`bcaff5eb818d6f65`) được ghi vào `scores.json` và RESULT.md, vì commit hash một mình không xác định được dữ liệu đã dùng nếu corpus thay đổi giữa hai lần chạy.

## Điều còn hạn chế

- **Hạn chế cụ thể:** n = 20 đủ để thấy một lỗi mất bằng chứng và một lỗi chunking, nhưng **không đủ để coi delta cỡ 0.01 là có ý nghĩa**. Average (−0.0101), faithfulness (−0.0182) và answer relevance (+0.0082) đều nằm trong khoảng nhiễu; chỉ context recall (−0.10) và context precision (+0.07) là đủ lớn để kết luận, và cả hai đều được chỉ số tất định xác nhận độc lập. Nhãn `retrieval_challenge` cũng là chủ đích thiết kế câu hỏi, không phải kết quả rút ra từ dữ liệu.
- **Nếu có thêm thời gian:** Đổi đầu tiên là chấm lại bằng judge mạnh hơn (hoặc chấm 2 lần lấy trung bình) rồi so `per_case.B.q02.metrics.faithfulness` trong `scores.json` — hiện delta faithfulness −0.0182 gần như do một mình q02 tạo ra, nên đây là chỗ rẻ nhất để bớt nhiễu khỏi kết luận. Chạy bằng `--score-only` để chấm lại đúng bộ câu trả lời cũ, không sinh lại.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Nguyễn Khánh Đô
