"""
Chatbot RAG về pháp luật lao động Việt Nam.

UI này chỉ là lớp hiển thị của pipeline: nó nhận query và top_k, gọi
generate_with_citation() rồi bày ra đúng những gì generation đã dùng — answer,
nguồn, retrieval method và score.

Nguồn hiển thị lấy từ `sources` của GenerationResult, không phải từ URL do model
sinh ra: nhãn [Document N] trong câu trả lời trỏ đúng vào `sources[N-1]`, nên
người đọc đối chiếu được từng khẳng định với chunk đã đưa vào context. Lịch sử
trong st.session_state vì vậy phải lưu cả sources và retrieval_source, không chỉ
text, nếu không các câu trả lời cũ sẽ mất phần nguồn sau lần rerun kế tiếp.

Chạy:
    streamlit run app.py
"""

import re

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import LLM_MODEL, LLM_PROVIDER, generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot — Pháp luật lao động",
    page_icon="⚖️",
    layout="wide",
)

# Nhãn model sinh ra trong answer, dùng để đánh dấu nguồn nào thật sự được trích.
CITATION_PATTERN = re.compile(r"\[Document\s+(\d+)\]", re.IGNORECASE)

RETRIEVAL_LABELS = {
    "hybrid": "Hybrid (dense + BM25, fuse bằng RRF)",
    "pageindex": "PageIndex fallback (dense score dưới ngưỡng)",
    "none": "Không có nguồn — safe refusal",
}

SAMPLE_QUESTIONS = [
    "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?",
    "Thời giờ làm việc bình thường tối đa bao nhiêu giờ một ngày?",
    "Người lao động được nghỉ hằng năm bao nhiêu ngày?",
]


def cited_indexes(answer: str) -> set[int]:
    """Số hiệu [Document N] xuất hiện trong answer."""
    return {int(match) for match in CITATION_PATTERN.findall(answer)}


def render_sources(answer: str, sources: list[dict], retrieval_source: str) -> None:
    """Hiển thị nguồn đã đưa vào context, đánh số khớp với nhãn trong answer."""
    st.caption(f"Retrieval: {RETRIEVAL_LABELS.get(retrieval_source, retrieval_source)}")

    if not sources:
        return

    cited = cited_indexes(answer)
    with st.expander(f"Nguồn đã dùng ({len(sources)} chunk)", expanded=True):
        for index, source in enumerate(sources, start=1):
            metadata = source["metadata"]
            badge = "✅ được trích dẫn" if index in cited else "· không được trích dẫn"
            st.markdown(f"**[Document {index}]** {metadata['title']} — {badge}")

            columns = st.columns(3)
            columns[0].metric("Score", f"{source['score']:.6f}")
            columns[1].metric("Method", source["retrieval_method"])
            columns[2].metric("Chunk", metadata.get("chunk_index", "—"))

            st.caption(f"File: `{metadata['source']}` · Chunk ID: `{source['id']}`")
            if metadata.get("url"):
                st.caption(f"Nguồn công khai: {metadata['url']}")
            st.text(source["content"])
            st.divider()

    missing = cited - set(range(1, len(sources) + 1))
    if missing:
        # Nhãn không có trong sources nghĩa là model tự bịa số hiệu; nói thẳng
        # thay vì để người đọc tưởng có một nguồn thứ N nào đó.
        st.warning(
            "Câu trả lời trích dẫn nhãn không có trong nguồn: "
            + ", ".join(f"[Document {index}]" for index in sorted(missing))
        )


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption(
        "Hỏi đáp pháp luật lao động Việt Nam trên Bộ luật Lao động 2019, "
        "Nghị định 145/2020, Nghị định 12/2022, Nghị quyết 17/2022 và các bài "
        "báo đã thu thập."
    )
    top_k = st.slider("Số chunks", 3, 10, 5)
    st.caption(f"Generator: `{LLM_PROVIDER}` · `{LLM_MODEL or 'default của provider'}`")

    st.divider()
    st.caption("Câu hỏi mẫu")
    for question in SAMPLE_QUESTIONS:
        st.caption(f"- {question}")

    st.divider()
    if st.button("Xoá lịch sử", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.title("Chatbot pháp luật lao động")
st.caption(
    "Mỗi câu trả lời chỉ dựa trên context đã truy xuất và trích dẫn theo nhãn "
    "[Document N]; số hiệu khớp với danh sách nguồn ngay dưới câu trả lời. "
    "Câu hỏi ngoài phạm vi tài liệu sẽ nhận được câu từ chối thay vì phỏng đoán."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message["content"],
                message.get("sources", []),
                message.get("retrieval_source", "none"),
            )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            try:
                result = generate_with_citation(query, top_k)
            except Exception as error:  # noqa: BLE001 — UI không được chết vì pipeline
                st.error(f"Pipeline lỗi: {type(error).__name__}: {error}")
                result = {
                    "answer": "Xin lỗi, hệ thống đang gặp sự cố khi xử lý câu hỏi này.",
                    "sources": [],
                    "retrieval_source": "none",
                }

        st.markdown(result["answer"])
        render_sources(result["answer"], result["sources"], result["retrieval_source"])

    # Lưu đủ ba field để lần rerun sau render lại được nguồn của câu trả lời cũ.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
