# Individual contribution report

## Thông tin

- Họ và tên: Mai Văn Trung
- Mã học viên: 2A202602513
- Nhóm: K4-L3B — chatbot RAG pháp luật lao động Việt Nam
- Repository/branch: https://github.com/quan05102k4/K4-L3B-RAG-Pipeline — `main`
- Vai trò: Thành viên — Generation và UI

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 10 — generation có citation | Viết `generate_with_citation()` trả đúng `GenerationResult` ba field; `reorder_for_llm()` chống lost-in-the-middle mà không mutate chunk; `format_context()` gắn title + source vào từng đoạn | `src/task10_generation.py` | Done |
| Đánh số citation | Viết `_citation_order()` — thứ tự đánh nhãn dùng chung cho context và `sources`, khoá phụ theo `id` để ổn định khi RRF hoà điểm | `src/task10_generation.py:_citation_order` | Done |
| Safe refusal | `SYSTEM_PROMPT` cấm suy đoán và bịa số điều luật; `_safe_refusal()` cho ba field cùng rơi về trạng thái không nguồn khi retrieve rỗng hoặc provider lỗi | `src/task10_generation.py`, `REFUSAL_ANSWER` | Done |
| Multi-provider dispatch | Map `OPENAI_COMPATIBLE` cho openai/groq dùng chung client, nhánh riêng cho gemini và anthropic; model và base_url đọc từ `.env` | `src/task10_generation.py`, `.env.example` | Done |
| Chuẩn hoá nhãn CJK | Phát hiện gpt-oss trên Groq trả `【Document 1】` dù prompt yêu cầu ASCII; thêm `CITATION_BRACKETS` dịch về `[ ]` để UI parse được | `src/task10_generation.py:CITATION_BRACKETS` | Done |
| Chatbot Streamlit | UI chat hiển thị answer, nguồn đã dùng, retrieval method, score và chunk ID; đánh dấu chunk nào thật sự được trích dẫn; cảnh báo khi model bịa nhãn không có trong `sources`; lưu đủ 3 field vào `session_state` để rerun không mất nguồn | `app.py` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Số hiệu `[Document N]` được tính bằng `_citation_order()` (score giảm dần, hoà điểm thì theo `id`) và dùng chung cho cả context lẫn `sources` — **không** đánh số theo vị trí chunk trong context.
   **Lý do/evidence:** `reorder_for_llm()` cố ý đổi *vị trí* chunk trong context để chunk mạnh nhất nằm đầu và cuối, còn `sources` phải giữ thứ tự score giảm dần theo contract. Nếu đánh nhãn theo vị trí trong context thì `[Document 3]` trong câu trả lời sẽ trỏ sang nguồn thứ ba mà UI hiển thị — hai chunk khác nhau, và người đọc đối chiếu citation sẽ ra sai nguồn. Ràng buộc `sources[N-1]` này được khoá bằng `test_reorder_is_non_mutating_and_context_contains_source`, và kiểm lại trực tiếp trong demo: `[Document 3]` → `bo_luat_lao_dong_2019.md::chunk-166` → Điều 107 trong `data/standardized/`.
   **Trade-off:** Thứ tự nhãn trong context không còn là thứ tự đọc (context bắt đầu bằng `[Document 1]` nhưng đoạn thứ hai có thể là `[Document 5]`). Tôi chấp nhận đánh đổi này vì người đọc đối chiếu nguồn ở UI, không đọc context thô.

2. **Quyết định:** Groq đi qua client `openai` với `base_url` riêng trong map `OPENAI_COMPATIBLE`, thay vì thêm một SDK thứ hai.
   **Lý do/evidence:** Groq nói cùng giao thức chat completions; gộp vào một nhánh giữ `call_llm()` chỉ có một đường code cho hai provider, và đổi provider không phải sửa code — chỉ sửa `LLM_PROVIDER` trong `.env`. `LLM_BASE_URL` cho phép trỏ sang gateway tương thích OpenAI khác mà vẫn không đụng code.
   **Trade-off:** Dùng chung đường code nghĩa là phải chịu các khác biệt hành vi của từng model ở tầng trên. Cụ thể là gpt-oss trả nhãn bằng ngoặc vuông CJK — tôi phải thêm một bước dịch ký tự ở `generate_with_citation()` chứ không có chỗ nào khác chặn được.

## Kiểm thử và kết quả

- **Test khoá hành vi phần tôi làm:** `test_reorder_is_non_mutating_and_context_contains_source` (reorder không mutate, context có title/source), `test_generation_result_validator_accepts_safe_refusal`, và `test_public_function_signatures_are_stable` khoá signature `generate_with_citation(query, top_k)`. Pass trong `pytest -q` (20/20).
- **Demo query đúng domain** (`python -m group_project.evaluation.demo --part in`): "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?" → `retrieval_source=hybrid`, 5 sources, answer tách đúng mức chung **200 giờ/năm** và ngoại lệ **300 giờ/năm**, mỗi mức dẫn nhãn riêng. 4/5 chunk được trích; `[Document 2]` không được trích nhưng vẫn hiển thị kèm nhãn "không được trích dẫn" thay vì bị giấu.
- **Demo query ngoài domain:** "Công thức nấu phở bò Hà Nội cần những nguyên liệu gì?" → answer từ chối, **không nhãn `[Document N]` nào** xuất hiện, cả 5 chunk đều mang nhãn "không được trích dẫn".
- **Kết quả trước/sau:** trước khi thêm `CITATION_BRACKETS`, UI hiển thị cảnh báo "trích dẫn nhãn không có trong nguồn" cho mọi câu trả lời của gpt-oss vì regex `\[Document N\]` không khớp `【Document 1】`; sau khi dịch ký tự, nhãn map đúng về `sources[N-1]` — quan sát lại được trong transcript demo.
- **Lỗi đã phát hiện và cách xử lý:** ban đầu `st.session_state` chỉ lưu text câu trả lời, nên sau mỗi lần rerun các câu trả lời cũ mất hết phần nguồn. Sửa bằng cách lưu đủ `content` + `sources` + `retrieval_source` cho mỗi message assistant.

## Điều còn hạn chế

- **Hạn chế cụ thể:** Việc từ chối ở query ngoài domain hiện **phụ thuộc vào model**, không phải một cổng tất định. Khi `best_dense_score` dưới ngưỡng mà PageIndex fallback không bật, `retrieve()` vẫn trả về 5 chunk (RRF luôn xếp hạng được một top-5 dù không chunk nào liên quan), nên nhánh `_safe_refusal()` không kích hoạt và thứ chặn câu trả lời bịa chỉ là `SYSTEM_PROMPT`. gpt-oss-120b từ chối đúng, nhưng một model yếu hơn về instruction-following có thể bịa mà ngưỡng 0.60 không chặn được.
- **Nếu có thêm thời gian:** Đổi đầu tiên là cho `generate_with_citation()` trả thẳng safe refusal khi `best_dense_score` dưới ngưỡng *và* fallback không trả về gì — biến việc từ chối thành hành vi của pipeline thay vì của prompt. Kiểm chứng bằng cách chạy lại demo query ngoài domain với `LLM_PROVIDER` khác và xem câu từ chối có còn giữ nguyên không.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Mai Văn Trung
