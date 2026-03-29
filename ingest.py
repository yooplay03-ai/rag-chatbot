"""
ingest.py — 문서를 청킹하고 ChromaDB에 저장하는 모듈
사용법: python ingest.py
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


# ── 설정 ──────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"          # ChromaDB 저장 경로 (로컬)
COLLECTION_NAME = "my_documents"    # 컬렉션 이름
CHUNK_SIZE = 500                     # 청크 당 최대 글자 수
CHUNK_OVERLAP = 50                   # 청크 간 겹치는 글자 수 (문맥 유지)
EMBED_MODEL = "all-MiniLM-L6-v2"    # SentenceTransformers 임베딩 모델


# ── ChromaDB 클라이언트 초기화 ──────────────────────────────────────────────
def get_collection():
    """ChromaDB 컬렉션을 가져오거나 새로 생성한다."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # SentenceTransformers 임베딩 함수 사용 (로컬, 무료)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}  # 유사도 계산 방식: 코사인
    )
    return collection


# ── 텍스트 청킹 ────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    텍스트를 일정 크기의 청크로 분할한다.
    overlap 만큼 이전 청크와 겹쳐서 문맥 단절을 줄인다.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # 다음 청크 시작점 (overlap 만큼 뒤로 이동)
    return chunks


# ── 데이터 추가 ────────────────────────────────────────────────────────────────
def add_document(text: str, doc_id: str, metadata: dict = None):
    """
    문서 하나를 청킹 후 ChromaDB에 저장한다.

    Args:
        text:     저장할 원본 텍스트
        doc_id:   문서 고유 ID (예: "wiki_python", "readme")
        metadata: 추가 정보 (예: {"source": "wikipedia", "date": "2024-01"})
    """
    collection = get_collection()
    chunks = chunk_text(text)

    if not chunks:
        print(f"[경고] 청크가 없습니다: {doc_id}")
        return

    # 각 청크에 고유 ID 부여: "doc_id_0", "doc_id_1", ...
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metas = [{**(metadata or {}), "doc_id": doc_id, "chunk_index": i}
             for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metas)
    print(f"[추가 완료] '{doc_id}' → {len(chunks)}개 청크 저장")


def add_documents_from_folder(folder_path: str):
    """
    폴더 내 .txt 파일을 모두 읽어 ChromaDB에 저장한다.
    파일명이 doc_id로 사용된다.
    """
    folder = Path(folder_path)
    txt_files = list(folder.glob("*.txt"))

    if not txt_files:
        print(f"[경고] '{folder_path}'에 .txt 파일이 없습니다.")
        return

    for file in txt_files:
        text = file.read_text(encoding="utf-8")
        doc_id = file.stem  # 확장자 제외 파일명
        add_document(text, doc_id=doc_id, metadata={"source": str(file)})


# ── 데이터 삭제 ────────────────────────────────────────────────────────────────
def delete_document(doc_id: str):
    """
    특정 doc_id에 해당하는 모든 청크를 ChromaDB에서 삭제한다.
    """
    collection = get_collection()

    # doc_id가 포함된 청크 ID 조회
    results = collection.get(where={"doc_id": doc_id})
    ids_to_delete = results["ids"]

    if not ids_to_delete:
        print(f"[경고] '{doc_id}' 문서를 찾을 수 없습니다.")
        return

    collection.delete(ids=ids_to_delete)
    print(f"[삭제 완료] '{doc_id}' → {len(ids_to_delete)}개 청크 삭제")


def delete_all():
    """컬렉션의 모든 데이터를 삭제한다. (주의: 되돌릴 수 없음)"""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    client.delete_collection(name=COLLECTION_NAME)
    print(f"[삭제 완료] 컬렉션 '{COLLECTION_NAME}' 전체 삭제")


# ── 조회 ───────────────────────────────────────────────────────────────────────
def list_documents():
    """저장된 모든 문서 목록과 청크 수를 출력한다."""
    collection = get_collection()
    results = collection.get()

    if not results["ids"]:
        print("저장된 문서가 없습니다.")
        return

    # doc_id 별로 청크 수 집계
    doc_counts: dict[str, int] = {}
    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id", "unknown")
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    print(f"\n── 저장된 문서 목록 (총 {len(results['ids'])}개 청크) ──")
    for doc_id, count in sorted(doc_counts.items()):
        print(f"  · {doc_id:30s}  {count}개 청크")


# ── 직접 실행 시 예시 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 샘플 데이터 추가
    sample_text = """
    RAG(Retrieval-Augmented Generation)는 검색 기반 생성 방식으로,
    외부 문서를 검색해 LLM의 답변 품질을 높이는 기법입니다.
    전통적인 LLM은 학습 데이터에 없는 최신 정보나 사내 문서를 모릅니다.
    RAG는 이 문제를 해결하기 위해 질문과 관련된 문서를 먼저 검색하고,
    그 내용을 프롬프트에 포함시켜 LLM이 정확한 답변을 생성하게 합니다.
    주요 구성요소는 문서 저장소, 임베딩 모델, 벡터 DB, LLM입니다.
    """
    add_document(sample_text, doc_id="rag_intro", metadata={"topic": "AI"})

    # 저장된 문서 확인
    list_documents()
