"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ────────────────────────────────────
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

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Build RAG chain ──────────────────────────────────────────────────────
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


# ── 4. Chạy RAG và thu thập kết quả ────────────────────────────────────────
def run_rag(chain, retriever, question: str) -> dict:
    # Lấy docs được truy xuất
    docs = retriever.invoke(question)

    # QUAN TRỌNG: contexts phải là list[str], không ghép thành một string
    contexts = [doc.page_content for doc in docs]

    # Lấy câu trả lời
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "contexts": contexts,   # List các string riêng lẻ
    }


def collect_rag_outputs(chain_v1, chain_v2, retriever):
    results_v1, results_v2 = [], []

    for i, qa in enumerate(QA_PAIRS, 1):
        question = qa["question"]
        reference = qa["reference"]
        print(f"  Đang xử lý câu {i}/{len(QA_PAIRS)}...", end="\r")

        out_v1 = run_rag(chain_v1, retriever, question)
        out_v2 = run_rag(chain_v2, retriever, question)

        results_v1.append({
            "question": question,
            "reference": reference,
            **out_v1,
        })
        results_v2.append({
            "question": question,
            "reference": reference,
            **out_v2,
        })

    return results_v1, results_v2


# ── 5. Tạo RAGAS EvaluationDataset ─────────────────────────────────────────
def build_ragas_dataset(results: list) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],           # Câu hỏi
            response=r["answer"],               # Câu trả lời của LLM
            retrieved_contexts=r["contexts"],   # List[str] — contexts truy xuất được
            reference=r["reference"],           # Đáp án chuẩn từ qa_pairs.py
        )
        for r in results
    ]
    return EvaluationDataset(samples=samples)


# ── 6. Chạy RAGAS Evaluation ───────────────────────────────────────────────
def run_ragas_eval(dataset: EvaluationDataset, llm_eval, emb_eval) -> dict:
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    result = evaluate(dataset, metrics=metrics, llm=llm_eval, embeddings=emb_eval)

    scores = {}
    for metric in metrics:
        name = metric.name
        values = [value for value in result[name] if value is not None]
        scores[name] = float(np.nanmean(values)) if values else float("nan")

    return scores


# ── 7. Main ────────────────────────────────────────────────────────────────
def main():
    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()
    llm_eval = get_llm()       # LLM dùng cho RAGAS evaluation
    emb_eval = get_embeddings()

    chain_v1 = build_chain(retriever, PROMPT_V1, llm)
    chain_v2 = build_chain(retriever, PROMPT_V2, llm)

    print("Thu thap outputs tu 2 prompt versions...")
    results_v1, results_v2 = collect_rag_outputs(chain_v1, chain_v2, retriever)

    print("\nDang danh gia V1 (co the mat 10-20 phut)...")
    scores_v1 = run_ragas_eval(build_ragas_dataset(results_v1), llm_eval, emb_eval)

    print("\nDang danh gia V2 (co the mat 10-20 phut)...")
    scores_v2 = run_ragas_eval(build_ragas_dataset(results_v2), llm_eval, emb_eval)

    # In bảng so sánh
    print("\n" + "="*60)
    print(f"{'Chi so':<30} {'V1':>10} {'V2':>10}")
    print("-"*60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        print(f"{metric:<30} {scores_v1[metric]:>10.4f} {scores_v2[metric]:>10.4f}")
    print("="*60)

    # Lưu báo cáo
    report = {"v1": scores_v1, "v2": scores_v2}
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nDa luu bao cao vao data/ragas_report.json")


if __name__ == "__main__":
    main()
