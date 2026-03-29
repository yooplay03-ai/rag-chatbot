"""
chatbot.py — RAG 방식으로 Gemini API를 호출하는 챗봇 모듈
사용법: python chatbot.py
"""

import os
from google import genai
import chromadb
from chromadb.utils import embedding_functions


# ── 설정 ──────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"
EMBED_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3
GEMINI_MODEL = "gemini-2.0-flash-lite"  # 무료 티어, 한도 넉넉


# ── ChromaDB 컬렉션 가져오기 ──────────────────────────────────────────────────
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


# ── 관련 문서 검색 ─────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "doc_id": meta.get("doc_id", "unknown"),
            "distance": round(dist, 4)
        })
    return chunks


# ── 프롬프트 구성 ──────────────────────────────────────────────────────────────
def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[출처: {c['doc_id']}]\n{c['text']}" for c in context_chunks
    )
    return f"""아래는 참고할 수 있는 문서입니다:

{context}

---

위 문서를 참고하여 다음 질문에 답해주세요.
문서에 없는 내용은 모른다고 말해주세요.

질문: {query}"""


# ── Gemini API 호출 ────────────────────────────────────────────────────────────
def ask_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "터미널에서: set GEMINI_API_KEY=AIza..."
        )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text


# ── RAG 파이프라인 ─────────────────────────────────────────────────────────────
def rag_query(query: str, verbose: bool = False) -> str:
    chunks = retrieve(query)

    if not chunks:
        return "관련 문서를 찾지 못했습니다. 먼저 ingest.py로 문서를 추가해주세요."

    if verbose:
        print("\n── 검색된 청크 ──")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. [{c['doc_id']}] 유사도: {1 - c['distance']:.2%}")
            print(f"     {c['text'][:80]}...")
        print()

    prompt = build_prompt(query, chunks)
    return ask_gemini(prompt)


# ── 대화형 CLI ─────────────────────────────────────────────────────────────────
def run_chat():
    print("=" * 50)
    print("  RAG 챗봇 (ChromaDB + Gemini)")
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
