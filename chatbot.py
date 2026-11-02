"""
chatbot.py — RAG 방식으로 Groq API를 호출하는 챗봇 모듈
사용법: python chatbot.py   (FastAPI 서버는 api.py가 이 모듈의 answer_question을 사용)
"""

import os
import time
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions


# ── 설정 ──────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"
EMBED_MODEL = "BAAI/bge-m3"
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile은 2026-08-16 Groq에서 지원 종료

TOP_K = 3
CANDIDATE_K = 7   # 10 → 7: 리랭킹 시간 30% 감소, 근거 조항 손실 없음 (bench_rerank.py 참고)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# ── 무거운 객체는 처음 한 번만 만들고 재사용 ─────────────────────────────────
# lru_cache: 같은 함수를 다시 부르면 처음 만든 객체를 그대로 돌려준다.
# 예전에는 질문마다 get_collection()이 bge-m3(약 2GB)를 새로 불러와 답변이 느렸다.
@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "터미널에서: set GROQ_API_KEY=gsk_..."
        )
    return Groq(api_key=api_key)


# ── 관련 문서 검색 + Reranking ─────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    return rerank(query, search_candidates(query), top_k)


def search_candidates(query: str) -> list[dict]:
    """1단계: bge-m3로 질문을 벡터로 바꾸고, ChromaDB에서 가까운 조항 CANDIDATE_K개를 찾는다."""
    collection = get_collection()

    # 실제 저장된 문서 수 확인 후 candidate_k 조정
    total = collection.count()
    n_results = min(CANDIDATE_K, total)
    if n_results == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        candidates.append({
            "text": doc,
            "doc_id": meta.get("doc_id", "unknown"),
            "label": meta.get("label", meta.get("doc_id", "unknown")),  # 예: 표준취업규칙 제33조(연차유급휴가)
            "distance": round(dist, 4)
        })
    return candidates


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
    """2단계: 리랭커가 질문과 조항을 함께 읽고 점수를 매겨 상위 top_k개만 남긴다."""
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = get_reranker().predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ── 프롬프트 구성 ──────────────────────────────────────────────────────────────
def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[출처: {c['label']}]\n{c['text']}" for c in context_chunks
    )
    return f"""아래는 참고할 수 있는 문서입니다:

{context}

---

위 문서를 참고하여 다음 질문에 답해주세요.
- 답변 끝에 근거가 된 조항을 [출처]에 적힌 이름 그대로 밝혀주세요. (예: 근거: 표준취업규칙 제33조(연차유급휴가))
- 문서에 없는 내용은 모른다고 말해주세요.
- 문서에 '00:00', '○○'처럼 비어 있는 값은 회사가 정해야 하는 부분이라고 안내해주세요.

질문: {query}"""


# ── Groq API 호출 ──────────────────────────────────────────────────────────────
def ask_groq(prompt: str) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ── RAG 파이프라인 ─────────────────────────────────────────────────────────────
NO_DOCS_MESSAGE = "관련 문서를 찾지 못했습니다. 먼저 ingest.py로 문서를 추가해주세요."


def answer_question(query: str) -> dict:
    """답변, 근거 청크, 단계별 걸린 시간(ms)을 함께 돌려준다. CLI와 API가 같이 쓴다."""
    ms = lambda a, b: round((b - a) * 1000, 1)
    t0 = time.perf_counter()
    candidates = search_candidates(query)
    t1 = time.perf_counter()
    chunks = rerank(query, candidates)
    t2 = time.perf_counter()
    if not chunks:
        return {"answer": NO_DOCS_MESSAGE, "sources": [],
                "timings_ms": {"search": ms(t0, t1), "rerank": ms(t1, t2), "llm": 0.0}}
    answer = ask_groq(build_prompt(query, chunks))
    t3 = time.perf_counter()
    return {
        "answer": answer,
        "sources": chunks,
        "timings_ms": {"search": ms(t0, t1), "rerank": ms(t1, t2), "llm": ms(t2, t3)},
    }


def rag_query(query: str, verbose: bool = False) -> str:
    result = answer_question(query)
    chunks = result["sources"]

    if not chunks:
        return result["answer"]

    if verbose:
        print("\n── 검색된 청크 (reranking 적용) ──")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. [{c['label']}] rerank 점수: {c['rerank_score']:.4f}")
            print(f"     {c['text'][:80]}...")
        t = result["timings_ms"]
        print(f"  (임베딩 검색 {t['search']:.0f}ms, 리랭킹 {t['rerank']:.0f}ms, 답변 생성 {t['llm']:.0f}ms)")
        print()

    return result["answer"]


# ── 대화형 CLI ─────────────────────────────────────────────────────────────────
def run_chat():
    print("=" * 50)
    print("  RAG 챗봇 (ChromaDB + Groq)")
    print("  종료: 'q' 또는 'quit' 입력")
    print("=" * 50)

    while True:
        try:
            query = input("\n질문: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("종료합니다.")
            break

        print("\n답변 생성 중...")
        try:
            answer = rag_query(query, verbose=True)
            print(f"\n답변:\n{answer}")
        except Exception as e:
            print(f"\n[오류] {e}")


if __name__ == "__main__":
    run_chat()