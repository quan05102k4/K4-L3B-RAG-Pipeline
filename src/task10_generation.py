"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.

Ba field của GenerationResult phải mô tả cùng một lần truy xuất: `sources` là
đúng những chunk đã đưa vào context, `retrieval_source` là đường đã sinh ra
chúng (hybrid hoặc pageindex), và `answer` chỉ được trích dẫn nhãn có trong
context. Khi không có chunk nào dùng được, cả ba cùng rơi về safe refusal —
answer từ chối, sources rỗng, retrieval_source "none".

Điểm dễ sai nhất của module là số thứ tự citation. reorder_for_llm() đổi *vị
trí* chunk trong context, còn `sources` phải giữ thứ tự score giảm dần theo
contract. Đánh số "Document N" theo vị trí trong context thì [Document 3] trong
câu trả lời sẽ trỏ sang nguồn thứ ba mà UI hiển thị — hai thứ khác nhau. Vì vậy
nhãn được tính bằng _citation_order(), thứ tự dùng chung cho cả context lẫn
`sources`, nên [Document N] luôn là sources[N-1].

Chạy:
    python -m src.task10_generation "câu hỏi của bạn"
"""

import os
import sys

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 2048

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# OpenAI và Groq nói cùng một giao thức chat completions; Groq chỉ khác base_url
# nên dùng chung client `openai` thay vì thêm một SDK nữa. Nhóm đang chạy key
# Groq (LLM_PROVIDER=groq trong .env); nhánh openai giữ nguyên cho máy khác.
OPENAI_COMPATIBLE = {
    "openai": {
        "key_name": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": "gpt-4.1-mini",
    },
    "groq": {
        "key_name": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
    },
}
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-5",
}

# Câu từ chối dùng chung cho mọi nhánh không có evidence, để UI và evaluation
# nhận ra safe refusal bằng đúng một chuỗi.
REFUSAL_ANSWER = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

# gpt-oss trên Groq trả nhãn bằng ngoặc vuông CJK 【Document 1】 dù prompt yêu cầu
# ASCII. Chuẩn hoá lại để UI parse được citation về đúng phần tử trong sources.
CITATION_BRACKETS = str.maketrans({"【": "[", "】": "]"})

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh.

Quy tắc trích dẫn:
- Trích dẫn bằng đúng nhãn [Document N] đứng đầu đoạn context mà bạn đã dùng.
- Nhãn viết bằng ngoặc vuông ASCII, đúng dạng [Document 1]; không dùng 【】.
- Mỗi khẳng định dẫn nhãn của đoạn chứa nó; không gộp nhiều đoạn vào một nhãn.
- Không bịa URL, số điều luật, mốc thời gian hay con số không có trong context.
- Nếu context không trả lời được câu hỏi, nói rõ là không xác minh được từ
  nguồn hiện có và dừng lại, không suy đoán thêm.
- Trả lời bằng tiếng Việt."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context.

    LLM đọc kỹ đầu và cuối context hơn phần giữa (lost-in-the-middle), nên chunk
    hạng 1 nằm đầu, hạng 2 nằm cuối, phần còn lại dồn vào giữa. Hàm trả về list
    mới và không sửa dict nào: ID và score phải đi tiếp nguyên vẹn sang
    `sources`, nếu không citation mất đường truy ngược về chunk.
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label.

    Nhãn `[Document N | Title: ... | Source: ...]` là thứ duy nhất LLM có để
    trích dẫn, và N tính bằng _citation_order() nên trỏ đúng `sources[N-1]` kể
    cả khi chunks truyền vào đã bị reorder.
    """
    if not chunks:
        return ""

    labels = {chunk["id"]: index for index, chunk in enumerate(_citation_order(chunks), 1)}

    parts = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        parts.append(
            f"[Document {labels[chunk['id']]} | Title: {metadata.get('title', '')} | "
            f"Source: {metadata.get('source', '')}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình.

    Ranh giới hàm là text thuần: mọi nhánh trả về một str đã strip, còn lỗi
    provider ném ra ngoài cho generate_with_citation đổi thành safe refusal.
    """
    provider = LLM_PROVIDER.strip().lower()

    if provider in OPENAI_COMPATIBLE:
        return _call_openai_compatible(provider, system_prompt, user_message)
    if provider == "gemini":
        return _call_gemini(system_prompt, user_message)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_message)

    raise ValueError(
        f"LLM_PROVIDER không hỗ trợ: {LLM_PROVIDER!r}. "
        "Dùng openai | groq | gemini | anthropic."
    )


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    if not query or not query.strip() or top_k <= 0:
        return _safe_refusal()

    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return _safe_refusal()

    # Đánh số citation một lần, dùng chung cho context và sources trả ra.
    ordered = _citation_order(chunks)
    context = format_context(reorder_for_llm(ordered))
    user_message = f"Context:\n{context}\n\nCâu hỏi: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:  # noqa: BLE001 — provider ngoài, UI không được chết
        print(
            f"[generate_with_citation] provider lỗi "
            f"({type(error).__name__}: {error}); trả safe refusal.",
            file=sys.stderr,
        )
        # Không giữ lại sources ở nhánh này: không có câu trả lời nào trích dẫn
        # chúng, mà hiển thị nguồn kèm một lời từ chối đúng là kiểu lệch ba
        # field mà contract muốn tránh.
        return _safe_refusal(
            f"{REFUSAL_ANSWER} (Không gọi được LLM provider: {type(error).__name__}.)"
        )

    if not answer.strip():
        return _safe_refusal()

    return {
        "answer": answer.strip().translate(CITATION_BRACKETS),
        "sources": ordered,
        "retrieval_source": _retrieval_source(ordered),
    }


def _citation_order(chunks: list[dict]) -> list[dict]:
    """Thứ tự đánh số citation: score giảm dần, hoà điểm thì theo id.

    RRF hoà điểm rất thường xuyên — chunk chỉ xuất hiện ở một danh sách và cùng
    rank thì nhận đúng cùng một score. Nếu chỉ sort theo score, thứ tự nhãn
    trong context và thứ tự nguồn UI hiển thị có thể hoán vị nhau ở những chunk
    hoà điểm. Khoá phụ `id` làm thứ tự ổn định dù nhận danh sách gốc hay danh
    sách đã reorder, và vẫn là score giảm dần nên `sources` đúng contract.
    """
    return sorted(chunks, key=lambda chunk: (-float(chunk["score"]), chunk["id"]))


def _retrieval_source(chunks: list[dict]) -> str:
    """Đường truy xuất đã sinh ra sources, theo enum của GenerationResult.

    retrieve() trả về pageindex result khi fallback bật, còn lại là hybrid. Enum
    RetrievalSource không có "dense"/"bm25" nên mọi nhánh không-pageindex quy về
    "hybrid" — đúng với đường mặc định use_reranking=True.
    """
    return "pageindex" if chunks[0]["retrieval_method"] == "pageindex" else "hybrid"


def _safe_refusal(answer: str = REFUSAL_ANSWER) -> dict:
    """GenerationResult khi không có evidence: ba field cùng rỗng nguồn."""
    return {"answer": answer, "sources": [], "retrieval_source": "none"}


def _require_key(key_name: str) -> str:
    key = os.getenv(key_name, "").strip()
    if not key:
        raise ValueError(f"Thiếu {key_name} trong .env cho LLM_PROVIDER={LLM_PROVIDER!r}.")
    return key


def _model_for(default_model: str) -> str:
    """LLM_MODEL trong .env thắng; để trống thì dùng default của provider."""
    return LLM_MODEL.strip() or default_model


def _call_openai_compatible(provider: str, system_prompt: str, user_message: str) -> str:
    from openai import OpenAI

    config = OPENAI_COMPATIBLE[provider]
    # LLM_BASE_URL để trỏ sang gateway tương thích OpenAI mà không phải sửa code.
    base_url = os.getenv("LLM_BASE_URL", "").strip() or config["base_url"]
    client = OpenAI(api_key=_require_key(config["key_name"]), base_url=base_url)

    response = client.chat.completions.create(
        model=_model_for(config["default_model"]),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


def _call_gemini(system_prompt: str, user_message: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=_model_for(DEFAULT_MODELS["gemini"]),
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    return (response.text or "").strip()


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=_model_for(DEFAULT_MODELS["anthropic"]),
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        # Anthropic khuyến nghị chỉnh temperature hoặc top_p, không đặt cả hai.
        temperature=TEMPERATURE,
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "Thời giờ làm việc bình thường tối đa bao nhiêu giờ một ngày?"

    result = generate_with_citation(question)

    print(f"Query: {question}")
    print(f"Provider: {LLM_PROVIDER} | retrieval_source: {result['retrieval_source']}\n")
    print(result["answer"])
    print()
    for index, source in enumerate(result["sources"], start=1):
        metadata = source["metadata"]
        print(
            f"[Document {index}] {metadata['title']} — {metadata['source']} "
            f"({source['retrieval_method']} {source['score']:.6f})"
        )
        print(f"   {source['id']}")
