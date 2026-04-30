"""
ingest.py — 문서를 청킹하고 ChromaDB에 저장하는 모듈
사용법: python ingest.py
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


# ── 설정 ──────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── ChromaDB 클라이언트 초기화 ──────────────────────────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ── 텍스트 청킹 ────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── 데이터 추가 ────────────────────────────────────────────────────────────────
def add_document(text: str, doc_id: str, metadata: dict = None):
    collection = get_collection()
    chunks = chunk_text(text)

    if not chunks:
        print(f"[경고] 청크가 없습니다: {doc_id}")
        return

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metas = [{**(metadata or {}), "doc_id": doc_id, "chunk_index": i}
             for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metas)
    print(f"[추가 완료] '{doc_id}' → {len(chunks)}개 청크 저장")


def add_documents_from_folder(folder_path: str):
    folder = Path(folder_path)
    txt_files = list(folder.glob("*.txt"))

    if not txt_files:
        print(f"[경고] '{folder_path}'에 .txt 파일이 없습니다.")
        return

    for file in txt_files:
        text = file.read_text(encoding="utf-8")
        doc_id = file.stem
        add_document(text, doc_id=doc_id, metadata={"source": str(file)})


# ── 데이터 삭제 ────────────────────────────────────────────────────────────────
def delete_document(doc_id: str):
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    ids_to_delete = results["ids"]

    if not ids_to_delete:
        print(f"[경고] '{doc_id}' 문서를 찾을 수 없습니다.")
        return

    collection.delete(ids=ids_to_delete)
    print(f"[삭제 완료] '{doc_id}' → {len(ids_to_delete)}개 청크 삭제")


def delete_all():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    client.delete_collection(name=COLLECTION_NAME)
    print(f"[삭제 완료] 컬렉션 '{COLLECTION_NAME}' 전체 삭제")


# ── 조회 ───────────────────────────────────────────────────────────────────────
def list_documents():
    collection = get_collection()
    results = collection.get()

    if not results["ids"]:
        print("저장된 문서가 없습니다.")
        return

    doc_counts: dict[str, int] = {}
    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id", "unknown")
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    print(f"\n── 저장된 문서 목록 (총 {len(results['ids'])}개 청크) ──")
    for doc_id, count in sorted(doc_counts.items()):
        print(f"  · {doc_id:30s}  {count}개 청크")


# ── 직접 실행 시 샘플 데이터 ──────────────────────────────────────────────────
if __name__ == "__main__":

    # 기존 데이터 초기화
    delete_all()

    # 1. RAG 소개
    add_document("""
    RAG(Retrieval-Augmented Generation)는 검색 기반 생성 방식으로,
    외부 문서를 검색해 LLM의 답변 품질을 높이는 기법입니다.
    전통적인 LLM은 학습 데이터에 없는 최신 정보나 사내 문서를 모릅니다.
    RAG는 이 문제를 해결하기 위해 질문과 관련된 문서를 먼저 검색하고,
    그 내용을 프롬프트에 포함시켜 LLM이 정확한 답변을 생성하게 합니다.
    주요 구성요소는 문서 저장소, 임베딩 모델, 벡터 DB, LLM입니다.
    RAG의 장점은 최신 정보 반영, 환각 감소, 출처 추적이 가능하다는 점입니다.
    """, doc_id="rag_intro", metadata={"topic": "AI"})

    # 2. 임베딩 모델 설명
    add_document("""
    임베딩(Embedding)은 텍스트를 숫자 벡터로 변환하는 기술입니다.
    의미가 비슷한 문장은 벡터 공간에서 가까운 위치에 놓이게 됩니다.
    RAG에서 임베딩 모델은 문서와 질문을 벡터로 변환해 유사도를 계산합니다.
    대표적인 임베딩 모델로는 all-MiniLM-L6-v2, bge-m3, text-embedding-3 등이 있습니다.
    한국어 RAG를 만들 때는 다국어를 지원하는 bge-m3 모델이 권장됩니다.
    임베딩 모델의 성능이 RAG 검색 품질을 크게 좌우합니다.
    벡터 차원이 클수록 정밀하지만 속도와 비용이 증가합니다.
    """, doc_id="embedding_intro", metadata={"topic": "AI"})

    # 3. 벡터 DB 설명
    add_document("""
    벡터 데이터베이스(Vector DB)는 임베딩 벡터를 저장하고 검색하는 데이터베이스입니다.
    일반 DB와 달리 벡터 간의 유사도를 기반으로 검색합니다.
    대표적인 벡터 DB로는 ChromaDB, Pinecone, Weaviate, Milvus 등이 있습니다.
    ChromaDB는 로컬에서 무료로 사용할 수 있어 개발 단계에서 많이 쓰입니다.
    코사인 유사도, 유클리드 거리 등의 방식으로 유사한 벡터를 찾습니다.
    HNSW 알고리즘을 사용해 대용량 벡터도 빠르게 검색할 수 있습니다.
    """, doc_id="vectordb_intro", metadata={"topic": "AI"})

    # 4. 청킹 설명
    add_document("""
    청킹(Chunking)은 긴 문서를 임베딩하기 좋은 크기로 잘라내는 작업입니다.
    LLM과 임베딩 모델은 한 번에 처리할 수 있는 텍스트 길이에 한계가 있습니다.
    청킹 전략에는 고정 크기 청킹, 재귀적 청킹, 시맨틱 청킹 등이 있습니다.
    chunk_overlap을 설정하면 청크 경계에서 문맥이 끊기는 것을 방지할 수 있습니다.
    일반적으로 chunk_size는 500~1000토큰, overlap은 10~20%가 권장됩니다.
    청킹 방식에 따라 RAG 검색 품질이 크게 달라집니다.
    재귀적 청킹은 문단과 문장 경계를 존중하며 자르기 때문에 가장 많이 사용됩니다.
    """, doc_id="chunking_intro", metadata={"topic": "AI"})

    # 5. LLM 설명
    add_document("""
    LLM(Large Language Model)은 대규모 텍스트 데이터로 학습된 언어 모델입니다.
    대표적인 LLM으로는 GPT-4, Claude, Gemini, LLaMA 등이 있습니다.
    LLM은 주어진 프롬프트를 바탕으로 자연스러운 텍스트를 생성합니다.
    RAG에서 LLM은 검색된 문서를 참고해 최종 답변을 생성하는 역할을 합니다.
    LLM의 단점은 학습 데이터 이후의 정보를 모르고 환각이 발생할 수 있다는 점입니다.
    Groq은 LLaMA 모델을 매우 빠른 속도로 실행할 수 있는 API 서비스입니다.
    오픈소스 LLM인 LLaMA는 Meta에서 개발했으며 무료로 사용 가능합니다.
    """, doc_id="llm_intro", metadata={"topic": "AI"})

    # 저장된 문서 확인
    list_documents()