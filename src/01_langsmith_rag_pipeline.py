"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    embeddings = get_embeddings()                  # Lấy embedding model từ factory
    docs = load_knowledge_base()                   # Load text từ data/knowledge_base.txt
    chunks = split_text(docs)                      # Chia thành chunks nhỏ hơn
    vectorstore = build_vectorstore(chunks, embeddings)  # Tạo FAISS index
    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý hữu ích. Chỉ trả lời dựa trên context được cung cấp. "
               "Nếu không tìm thấy thông tin, hãy nói 'Tôi không tìm thấy thông tin này'.\n\n"
               "Context:\n{context}"),
    ("human", "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
from langsmith import traceable

@traceable(name="rag-query")
def ask(chain, question: str) -> str:
    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("Đang khởi tạo vector store...")
    vectorstore = setup_vectorstore()

    print("Đang xây dựng RAG chain...")
    chain = build_rag_chain(vectorstore)

    print(f"Đang chạy {len(SAMPLE_QUESTIONS)} câu hỏi qua RAG pipeline...")
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}] Q: {question[:60]}...")
        print(f"      A: {answer[:80]}...\n")

    print("Hoàn thành! Kiểm tra traces tại https://smith.langchain.com")


if __name__ == "__main__":
    main()
