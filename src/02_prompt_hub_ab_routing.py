"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "nguyen-duy-khanh-prompt-v1"   # ví dụ: "nguyen-rag-v1"
PROMPT_V2_NAME = "nguyen-duy-khanh-prompt-v2"   # ví dụ: "nguyen-rag-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1: phong cách ngắn gọn, thân thiện.
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn, rõ ràng dựa trên context. "
    "Nếu không có thông tin, hãy nói thẳng là không biết."
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1 + "\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

# V2: phong cách chuyên nghiệp, có cấu trúc.
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Khi trả lời, hãy: "
    "1) Tóm tắt câu trả lời chính, "
    "2) Trích dẫn nguồn từ context, "
    "3) Nêu rõ mức độ chắc chắn của câu trả lời. "
    "Luôn dựa trên dữ liệu được cung cấp, không suy đoán thêm."
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2 + "\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client=None):
    client = client or Client(api_key=config.LANGSMITH_API_KEY)

    client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1,
                       description="V1: Phong cách ngắn gọn, thân thiện")
    client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2,
                       description="V2: Phong cách chuyên nghiệp, có cấu trúc")
    print(f"Da push thanh cong: {PROMPT_V1_NAME} va {PROMPT_V2_NAME}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client=None):
    client = client or Client(api_key=config.LANGSMITH_API_KEY)
    prompt_v1 = client.pull_prompt(PROMPT_V1_NAME)
    prompt_v2 = client.pull_prompt(PROMPT_V2_NAME)
    return prompt_v1, prompt_v2


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return "v1" if h % 2 == 0 else "v2"


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(chain, question: str, version_label: str) -> dict:
    answer = chain.invoke(question)
    return {
        "question": question,
        "answer": answer,
        "version": version_label,
    }


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Build RAG chains ─────────────────────────────────────────────────────
def build_chain(retriever, prompt, llm):
    """Compose retrieval, prompt rendering, LLM invocation, and parsing."""

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ── 9. Main ────────────────────────────────────────────────────────────────
def main():
    vectorstore = setup_vectorstore()
    prompt_v1, prompt_v2 = pull_prompts_from_hub()

    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain_v1 = build_chain(retriever, prompt_v1, llm)
    chain_v2 = build_chain(retriever, prompt_v2, llm)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id = f"req-{i:03d}"
        version = get_prompt_version(request_id)
        chain = chain_v1 if version == "v1" else chain_v2

        result = ask_ab(chain, question, version)
        print(f"[{request_id}] [{result['version']}] Q: {question[:50]}...")
        print(f"             A: {result['answer'][:70]}...\n")


if __name__ == "__main__":
    main()
