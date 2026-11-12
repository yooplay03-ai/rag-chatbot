"""
rag-chatbot 검색 성능 측정 스크립트
====================================
목적: 실제 챗봇의 ingest.py / chatbot.py 로직을 그대로 재현해서
  1) 검색+리랭킹 응답 속도 (Groq LLM 호출 제외 — API 키 불필요)
  2) 검색 정확도 (Top-1 정확도, Top-3 재현율)
를 측정합니다. Groq API는 호출하지 않으므로 GROQ_API_KEY가 없어도 실행됩니다.

사용법:
  1) 이 파일을 rag_chatbot 폴더 안에 넣습니다 (ingest.py, chatbot.py와 같은 위치).
  2) 기존에 쓰던 가상환경을 활성화한 상태에서 실행:
       python bench_measure.py
  3) 콘솔에 출력되는 결과 + 같은 폴더에 생성되는 bench_result.txt 내용을 알려주세요.

주의: 기존 chroma_db는 건드리지 않고, ./bench_chroma_db 라는 별도 폴더에
      새로 색인해서 측정하므로 원래 쓰던 데이터에는 영향이 없습니다.
"""
import time
import statistics
import sys

try:
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import CrossEncoder
except ImportError as e:
    print(f"[오류] 필요한 패키지가 없습니다: {e}")
    print("실행: pip install chromadb sentence-transformers")
    sys.exit(1)

# ── 설정 (실제 ingest.py / chatbot.py와 동일) ─────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"
CANDIDATE_K = 10
TOP_K = 3
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CHROMA_PATH = "./bench_chroma_db"
COLLECTION_NAME = "bench_documents"

# ── 실제 ingest.py __main__ 블록의 5개 샘플 문서 그대로 사용 ─────────────
DOCS = {
    "rag_intro": """
    RAG(Retrieval-Augmented Generation)는 검색 기반 생성 방식으로,
    외부 문서를 검색해 LLM의 답변 품질을 높이는 기법입니다.
    전통적인 LLM은 학습 데이터에 없는 최신 정보나 사내 문서를 모릅니다.
    RAG는 이 문제를 해결하기 위해 질문과 관련된 문서를 먼저 검색하고,
    그 내용을 프롬프트에 포함시켜 LLM이 정확한 답변을 생성하게 합니다.
    주요 구성요소는 문서 저장소, 임베딩 모델, 벡터 DB, LLM입니다.
    RAG의 장점은 최신 정보 반영, 환각 감소, 출처 추적이 가능하다는 점입니다.
    """,
    "embedding_intro": """
    임베딩(Embedding)은 텍스트를 숫자 벡터로 변환하는 기술입니다.
    의미가 비슷한 문장은 벡터 공간에서 가까운 위치에 놓이게 됩니다.
    RAG에서 임베딩 모델은 문서와 질문을 벡터로 변환해 유사도를 계산합니다.
    대표적인 임베딩 모델로는 all-MiniLM-L6-v2, bge-m3, text-embedding-3 등이 있습니다.
    한국어 RAG를 만들 때는 다국어를 지원하는 bge-m3 모델이 권장됩니다.
    임베딩 모델의 성능이 RAG 검색 품질을 크게 좌우합니다.
    벡터 차원이 클수록 정밀하지만 속도와 비용이 증가합니다.
    """,
    "vectordb_intro": """
    벡터 데이터베이스(Vector DB)는 임베딩 벡터를 저장하고 검색하는 데이터베이스입니다.
    일반 DB와 달리 벡터 간의 유사도를 기반으로 검색합니다.
    대표적인 벡터 DB로는 ChromaDB, Pinecone, Weaviate, Milvus 등이 있습니다.
    ChromaDB는 로컬에서 무료로 사용할 수 있어 개발 단계에서 많이 쓰입니다.
    코사인 유사도, 유클리드 거리 등의 방식으로 유사한 벡터를 찾습니다.
    HNSW 알고리즘을 사용해 대용량 벡터도 빠르게 검색할 수 있습니다.
    """,
    "chunking_intro": """
    청킹(Chunking)은 긴 문서를 임베딩하기 좋은 크기로 잘라내는 작업입니다.
    LLM과 임베딩 모델은 한 번에 처리할 수 있는 텍스트 길이에 한계가 있습니다.
    청킹 전략에는 고정 크기 청킹, 재귀적 청킹, 시맨틱 청킹 등이 있습니다.
    chunk_overlap을 설정하면 청크 경계에서 문맥이 끊기는 것을 방지할 수 있습니다.
    일반적으로 chunk_size는 500~1000토큰, overlap은 10~20%가 권장됩니다.
    청킹 방식에 따라 RAG 검색 품질이 크게 달라집니다.
    재귀적 청킹은 문단과 문장 경계를 존중하며 자르기 때문에 가장 많이 사용됩니다.
    """,
    "llm_intro": """
    LLM(Large Language Model)은 대규모 텍스트 데이터로 학습된 언어 모델입니다.
    대표적인 LLM으로는 GPT-4, Claude, Gemini, LLaMA 등이 있습니다.
    LLM은 주어진 프롬프트를 바탕으로 자연스러운 텍스트를 생성합니다.
    RAG에서 LLM은 검색된 문서를 참고해 최종 답변을 생성하는 역할을 합니다.
    LLM의 단점은 학습 데이터 이후의 정보를 모르고 환각이 발생할 수 있다는 점입니다.
    Groq은 LLaMA 모델을 매우 빠른 속도로 실행할 수 있는 API 서비스입니다.
    오픈소스 LLM인 LLaMA는 Meta에서 개발했으며 무료로 사용 가능합니다.
    """,
}

# 문서당 3개씩, 실제 사용자가 물어볼 법한 자연스러운 질문 (정답 doc_id 포함)
TEST_QUERIES = [
    ("RAG가 정확히 뭐고 왜 쓰는거야?", "rag_intro"),
    ("검색 증강 생성 방식의 장점이 궁금해", "rag_intro"),
    ("LLM이 최신 정보를 모르는 문제를 어떻게 해결해?", "rag_intro"),
    ("임베딩이란 개념을 쉽게 설명해줘", "embedding_intro"),
    ("문장을 벡터로 바꾸는 이유가 뭐야?", "embedding_intro"),
    ("한국어에 적합한 임베딩 모델 추천해줘", "embedding_intro"),
    ("벡터 데이터베이스는 일반 DB랑 뭐가 달라?", "vectordb_intro"),
    ("ChromaDB가 뭐고 왜 많이 쓰여?", "vectordb_intro"),
    ("HNSW 알고리즘이 뭐하는거야?", "vectordb_intro"),
    ("긴 문서를 나눠서 저장하는 이유가 뭐야?", "chunking_intro"),
    ("chunk_overlap을 설정하는 이유는?", "chunking_intro"),
    ("청킹 전략에는 어떤 종류가 있어?", "chunking_intro"),
    ("LLM이 뭔지 설명해줘", "llm_intro"),
    ("Groq이 뭐하는 서비스야?", "llm_intro"),
    ("LLM의 단점이 뭐야?", "llm_intro"),
]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
    )
    total_chunks = 0
    for doc_id, text in DOCS.items():
        chunks = chunk_text(text)
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids, metadatas=metas)
        total_chunks += len(chunks)
    print(f"[색인 완료] {total_chunks}개 청크, {len(DOCS)}개 문서")
    return collection


def retrieve(collection, reranker, query, top_k=TOP_K):
    total = collection.count()
    n_results = min(CANDIDATE_K, total)
    results = collection.query(query_texts=[query], n_results=n_results,
                                include=["documents", "metadatas", "distances"])
    candidates = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        candidates.append({"text": doc, "doc_id": meta.get("doc_id", "unknown"), "distance": round(dist, 4)})
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = round(float(score), 4)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("색인 구축 중...")
    collection = build_index()
    lines[-1] = f"[색인 완료]"

    log(f"리랭커 로딩 중 ({RERANK_MODEL})... (최초 실행 시 모델 다운로드로 몇 분 걸릴 수 있습니다)")
    reranker = CrossEncoder(RERANK_MODEL)

    # 워밍업 (모델 최초 로딩 비용은 latency 측정에서 제외)
    _ = retrieve(collection, reranker, "워밍업 쿼리입니다")

    latencies = []
    top1_correct = 0
    top3_correct = 0
    per_query_results = []

    for query, gold_doc_id in TEST_QUERIES:
        t0 = time.perf_counter()
        chunks = retrieve(collection, reranker, query)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        retrieved_doc_ids = [c["doc_id"] for c in chunks]
        is_top1 = retrieved_doc_ids[0] == gold_doc_id if retrieved_doc_ids else False
        is_top3 = gold_doc_id in retrieved_doc_ids

        top1_correct += int(is_top1)
        top3_correct += int(is_top3)

        per_query_results.append((query, gold_doc_id, retrieved_doc_ids, latency_ms, is_top1, is_top3))

    n = len(TEST_QUERIES)
    log("\n=== 개별 쿼리 결과 ===")
    for query, gold, retrieved, lat, is_top1, is_top3 in per_query_results:
        mark = "O" if is_top1 else ("△" if is_top3 else "X")
        top1 = retrieved[0] if retrieved else None
        log(f"[{mark}] '{query}' -> top1={top1} (정답={gold}) {lat:.1f}ms")

    log("\n=== 요약 ===")
    log(f"쿼리 수: {n}")
    log(f"Top-1 정확도: {top1_correct}/{n} = {top1_correct/n*100:.1f}%")
    log(f"Top-3 재현율(Recall@3): {top3_correct}/{n} = {top3_correct/n*100:.1f}%")
    log(f"평균 응답 속도(검색+리랭킹, LLM 제외): {statistics.mean(latencies):.1f}ms")
    log(f"중앙값 응답 속도: {statistics.median(latencies):.1f}ms")
    log(f"최소/최대: {min(latencies):.1f}ms / {max(latencies):.1f}ms")

    with open("bench_result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("\n(결과가 bench_result.txt 파일로도 저장되었습니다)")


if __name__ == "__main__":
    main()
