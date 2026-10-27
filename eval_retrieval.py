"""
eval_retrieval.py — RAG 검색 품질 평가 (KorQuAD 1.0 기반)

질문마다 "정답이 들어 있는 문단"을 이미 알고 있는 KorQuAD 데이터로,
우리 검색기가 그 문단을 몇 번째로 찾아오는지 잰다. LLM은 부르지 않는다.

사용법:
    python eval_retrieval.py                           # 기본 설정 (MiniLM, 리랭킹 끔)
    python eval_retrieval.py --embed BAAI/bge-m3       # 임베딩 모델 바꿔서
    python eval_retrieval.py --rerank                  # 리랭킹 켜고
    python eval_retrieval.py --n-docs 100              # 빨리 돌려 보고 싶을 때
결과는 results.csv에 한 줄씩 쌓이고, 틀린 질문은 failures.csv에 저장된다.
"""

import argparse
import csv
import time
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

from ingest import chunk_text  # 챗봇과 똑같은 청킹 함수를 써야 평가가 의미 있다

KORQUAD_URL = (
    "https://huggingface.co/datasets/KorQuAD/squad_kor_v1/resolve/main/"
    "squad_kor_v1/validation-00000-of-00001.parquet"
)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
K = 10  # 검색 후보 수 (chatbot.py의 CANDIDATE_K와 같게)


# ── 1. 평가 데이터 준비 ──────────────────────────────────────────────────────
def load_korquad(n_docs: int, seed: int = 42):
    """문단 n_docs개와, 문단마다 질문 1개씩을 뽑는다."""
    df = pd.read_parquet(KORQUAD_URL)
    df = df.drop_duplicates(subset="context")  # 문단 하나에 질문 하나만 남김
    df = df.sample(n=min(n_docs, len(df)), random_state=seed).reset_index(drop=True)
    df["doc_id"] = [f"doc{i}" for i in range(len(df))]
    docs = df[["doc_id", "title", "context"]]
    questions = df[["doc_id", "question"]]
    return docs, questions


# ── 2. 벡터 DB에 넣기 ─────────────────────────────────────────────────────────
def get_embed_fn(model_name: str):
    import torch  # sentence-transformers를 설치하면 같이 깔린다
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU가 있으면 GPU로
    print(f"임베딩 계산 장치: {device}")
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name, device=device)


def build_index(docs: pd.DataFrame, embed_fn, chunk_size: int, overlap: int):
    """평가용 컬렉션을 메모리에 따로 만든다 (챗봇의 chroma_db는 건드리지 않음)."""
    client = chromadb.EphemeralClient()
    name = f"eval_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=name, embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
    )
    ids, texts, metas = [], [], []
    for row in docs.itertuples():
        for i, chunk in enumerate(chunk_text(row.context, chunk_size, overlap)):
            ids.append(f"{row.doc_id}_{i}")
            texts.append(chunk)
            metas.append({"doc_id": row.doc_id})
    for start in range(0, len(ids), 256):  # 한 번에 너무 많이 넣지 않게 나눠서
        end = start + 256
        collection.add(ids=ids[start:end], documents=texts[start:end], metadatas=metas[start:end])
    return collection


# ── 3. 검색 (+ 리랭킹) ───────────────────────────────────────────────────────
def search_all(collection, questions: list[str], k: int, reranker=None):
    """질문마다 검색된 청크의 (doc_id, 텍스트) 목록을 순서대로 돌려준다."""
    k = min(k, collection.count())
    results = []
    for start in range(0, len(questions), 64):
        batch = questions[start:start + 64]
        res = collection.query(query_texts=batch, n_results=k, include=["documents", "metadatas"])
        for q, docs, metas in zip(batch, res["documents"], res["metadatas"]):
            hits = [(m["doc_id"], d) for m, d in zip(metas, docs)]
            if reranker is not None:
                scores = reranker.predict([[q, d] for _, d in hits])
                hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]
            results.append(hits)
    return results


# ── 4. 점수 계산 ──────────────────────────────────────────────────────────────
def first_rank(hits, answer_doc_id):
    """정답 문단의 청크가 처음 나온 순위 (1부터). 못 찾으면 None."""
    for rank, (doc_id, _) in enumerate(hits, start=1):
        if doc_id == answer_doc_id:
            return rank
    return None


def score(ranks: list):
    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return {"hit@1": hit(1), "hit@3": hit(3), "hit@10": hit(10), "mrr": mrr}


# ── 5. 실행 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", default="all-MiniLM-L6-v2")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--n-docs", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    print("데이터 불러오는 중...")
    docs, questions = load_korquad(args.n_docs)

    print(f"색인 만드는 중... (임베딩: {args.embed})")
    t0 = time.perf_counter()
    collection = build_index(docs, get_embed_fn(args.embed), args.chunk_size, args.overlap)
    index_sec = time.perf_counter() - t0

    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL)

    print("검색 중...")
    t0 = time.perf_counter()
    all_hits = search_all(collection, questions["question"].tolist(), K, reranker)
    ms_per_q = (time.perf_counter() - t0) * 1000 / len(questions)

    ranks = [first_rank(h, d) for h, d in zip(all_hits, questions["doc_id"])]
    result = score(ranks)

    print(f"\n문단 {len(docs)}개 / 청크 {collection.count()}개 / 질문 {len(questions)}개")
    for name, value in result.items():
        print(f"  {name:7s} {value:.3f}")
    print(f"  질문당 검색 시간 {ms_per_q:.1f} ms (색인 {index_sec:.0f}초)")

    # 실험 기록 남기기: 실험할 때마다 한 줄씩 쌓인다
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "embed": args.embed, "rerank": args.rerank, "n_docs": len(docs),
        "chunk_size": args.chunk_size, "overlap": args.overlap,
        **{k: round(v, 3) for k, v in result.items()},
        "ms_per_q": round(ms_per_q, 1),
    }
    out = Path("results.csv")
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    # 틀린 질문 모아 두기: 왜 틀렸는지 읽어 보는 게 다음 개선의 출발점
    fails = [
        {"question": q, "answer_doc": d, "top1_doc": h[0][0] if h else "", "top1_text": h[0][1][:100] if h else ""}
        for q, d, h, r in zip(questions["question"], questions["doc_id"], all_hits, ranks)
        if r is None or r > 3
    ]
    pd.DataFrame(fails).to_csv("failures.csv", index=False, encoding="utf-8-sig")
    print(f"\nresults.csv에 기록, 상위 3개 안에 정답이 없던 질문 {len(fails)}개는 failures.csv에 저장")


if __name__ == "__main__":
    main()
