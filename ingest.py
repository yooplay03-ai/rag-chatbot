"""
ingest.py — 문서를 청킹하고 ChromaDB에 저장하는 모듈
사용법: python ingest.py   (docs 폴더의 .hwp / .txt 문서를 모두 다시 넣는다)
"""

import json
import os
import re
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError
from pathlib import Path

from hwp_loader import read_hwp_text
from rules_parser import apply_profile, parse_articles


# ── 설정 ──────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_METHOD = "sentence"   # "fixed": 글자 수로 자르기 / "sentence": 문장 경계에서 자르기 (README 실험 2 근거)
MIN_CHUNK_LEN = 100         # sentence 방식에서 이보다 짧은 꼬리는 앞 청크에 합침
EMBED_MODEL = "BAAI/bge-m3"
DOCS_DIR = "./docs"
PROFILE_PATH = "./company_profile.json"   # 표준안 빈칸을 채울 가상 회사 설정 (없으면 원본 그대로)
ARTICLE_MAX_LEN = 800       # 조항이 이보다 길면 문장 단위로 나눔


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


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")  # 마침표·물음표·느낌표 뒤 공백, 또는 줄바꿈


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def chunk_by_sentence(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap_sents: int = 1, min_len: int = MIN_CHUNK_LEN) -> list[str]:
    """문장을 chunk_size 안에서 최대한 모아 청크를 만든다.
    - 문장 중간에서 자르지 않는다 (한 문장이 chunk_size보다 길 때만 예외)
    - 앞 청크의 마지막 문장 overlap_sents개를 다음 청크 앞에 겹쳐 문맥을 잇는다
    - 마지막에 남은 꼬리가 min_len보다 짧으면 따로 두지 않고 앞 청크에 붙인다
    """
    pieces = []
    for sent in split_sentences(text):
        if len(sent) > chunk_size:   # 너무 긴 문장은 어쩔 수 없이 글자 수로 자름
            pieces.extend(chunk_text(sent, chunk_size, CHUNK_OVERLAP))
        else:
            pieces.append(sent)

    chunks, cur, n_new = [], [], 0   # n_new: cur 안에서 겹침이 아닌 새 문장 수
    for sent in pieces:
        if n_new > 0 and len(" ".join(cur + [sent])) > chunk_size:
            chunks.append(" ".join(cur))
            tail = cur[-overlap_sents:] if overlap_sents else []
            # 겹칠 문장이 너무 길면 청크가 커지므로 겹치지 않는다
            cur = tail if len(" ".join(tail)) <= chunk_size // 2 else []
            n_new = 0
        cur.append(sent)
        n_new += 1

    if n_new > 0:
        new_part = " ".join(cur[-n_new:])
        if chunks and len(new_part) < min_len:
            chunks[-1] += " " + new_part     # 짧은 꼬리는 앞 청크에 합치기
        else:
            chunks.append(" ".join(cur))
    return chunks


def make_chunks(text: str, method: str = CHUNK_METHOD) -> list[str]:
    if method == "sentence":
        return chunk_by_sentence(text)
    return chunk_text(text)


# ── 데이터 추가 ────────────────────────────────────────────────────────────────
def add_document(text: str, doc_id: str, metadata: dict = None):
    collection = get_collection()
    chunks = make_chunks(text)

    if not chunks:
        print(f"[경고] 청크가 없습니다: {doc_id}")
        return

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metas = [{**(metadata or {}), "doc_id": doc_id, "chunk_index": i}
             for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metas)
    print(f"[추가 완료] '{doc_id}' → {len(chunks)}개 청크 저장")


def add_rules_document(path: str, doc_name: str, profile: dict = None):
    """취업규칙 같은 규정 문서를 조항 단위로 넣는다. 청크마다 조항 정보를 메타데이터로 붙인다."""
    text = read_hwp_text(path) if path.endswith(".hwp") else Path(path).read_text(encoding="utf-8")
    articles = parse_articles(text, doc_name)
    if profile:
        articles = apply_profile(articles, profile)
        doc_name = profile.get("doc_name", doc_name)

    ids, chunks, metas = [], [], []
    for i, a in enumerate(articles):
        if len(a["text"]) <= ARTICLE_MAX_LEN:
            parts = [a["text"]]
        else:
            parts = chunk_by_sentence(a["text"], ARTICLE_MAX_LEN)
        for j, part in enumerate(parts):
            if j > 0:   # 나뉜 뒷부분에도 어느 조항인지 적어 둔다
                part = f"{a['article']}({a['title']}) (이어서) {part}"
            ids.append(f"{doc_name}_{i:03d}_{j}")
            chunks.append(part)
            metas.append({
                "doc_id": doc_name, "label": a["label"], "article": a["article"],
                "title": a["title"], "chapter": a["chapter"], "chunk_index": j,
            })

    collection = get_collection()
    for start in range(0, len(ids), 64):
        end = start + 64
        collection.add(ids=ids[start:end], documents=chunks[start:end], metadatas=metas[start:end])
    print(f"[추가 완료] '{doc_name}' → 조항 {len(articles)}개, {len(chunks)}개 청크 저장")


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
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"[삭제 완료] 컬렉션 '{COLLECTION_NAME}' 전체 삭제")
    except NotFoundError:
        # 처음 실행하는 환경처럼 컬렉션이 아직 없으면 지울 것이 없으므로 넘어간다
        print(f"[안내] 컬렉션 '{COLLECTION_NAME}'이 없어서 삭제를 건너뜁니다")


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


# ── 직접 실행: docs 폴더 문서 넣기 ──────────────────────────────────────────────────
if __name__ == "__main__":

    # 기존 데이터 초기화
    delete_all()

    folder = Path(DOCS_DIR)
    files = sorted(folder.glob("*.hwp")) + sorted(folder.glob("*.txt"))
    if not files:
        print(f"[안내] '{DOCS_DIR}' 폴더에 .hwp 또는 .txt 문서를 넣어 주세요.")

    profile = None
    if Path(PROFILE_PATH).exists():
        profile = json.loads(Path(PROFILE_PATH).read_text(encoding="utf-8"))
        print(f"[안내] 회사 설정 적용: {profile.get('company_name')} ({len(profile.get('edits', []))}개 문장 교체)")

    for file in files:
        name = re.sub(r"\(.*?\)", "", file.stem).strip()   # "표준취업규칙(2026년, 배포)" → "표준취업규칙"
        if file.suffix == ".hwp":
            add_rules_document(str(file), doc_name=name, profile=profile)
        else:
            add_document(file.read_text(encoding="utf-8"), doc_id=name, metadata={"source": str(file)})

    # 저장된 문서 확인
    list_documents()
