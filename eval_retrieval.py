"""
eval_retrieval.py — RAG 검색 품질 평가 (KorQuAD 1.0 기반)

질문마다 "정답이 들어 있는 문단"을 이미 알고 있는 KorQuAD 데이터로,
우리 검색기가 그 문단을 몇 번째로 찾아오는지 잰다. LLM은 부르지 않는다.

사용법:
    python eval_retrieval.py                           # 기본 설정 (MiniLM, 리랭킹 끔)
    python eval_retrieval.py --embed BAAI/bge-m3       # 임베딩 모델 바꿔서
    python eval_retrieval.py --rerank                  # 리랭킹 켜고
    python eval_retrieval.py --n-docs 100              # 빨리 돌려 보고 싶을 때
    python eval_retrieval.py --chunker sentence        # 문장 단위 청킹으로
    python eval_retrieval.py --n-docs 5000             # 검증 데이터의 문단 전체로 (더 어려운 평가)
결과는 results.csv에 한 줄씩 쌓이고, 틀린 질문은 failures_<설정>.csv에 저장된다.
--rerank를 켜면 같은 후보로 리랭킹 전과 후를 함께 기록한다.
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

from ingest import chunk_by_sentence, chunk_text  # 챗봇과 똑같은 청킹 함수를 써야 평가가 의미 있다

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


def make_chunks(text: str, method: str, chunk_size: int, overlap: int) -> list[str]:
    if method == "sentence":
        return chunk_by_sentence(text, chunk_size)
    return chunk_text(text, chunk_size, overlap)


def build_index(docs: pd.DataFrame, embed_fn, method: str, chunk_size: int, overlap: int):
    """평가용 컬렉션을 메모리에 따로 만든다 (챗봇의 chroma_db는 건드리지 않음)."""
    client = chromadb.EphemeralClient()
    name = f"eval_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=name, embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
    )
    ids, texts, metas = [], [], []
    for row in docs.itertuples():
        for i, chunk in enumerate(make_chunks(row.context, method, chunk_size, overlap)):
            ids.append(f"{row.doc_id}_{i}")
            texts.append(chunk)
            metas.append({"doc_id": row.doc_id})
    for start in range(0, len(ids), 256):  # 한 번에 너무 많이 넣지 않게 나눠서
        end = start + 256
        collection.add(ids=ids[start:end], documents=texts[start:end], metadatas=metas[start:end])
    lengths = [len(t) for t in texts]
    stats = {
        "n_chunks": len(texts),
        "avg_len": round(sum(lengths) / len(lengths)),
        "short_chunks": sum(1 for n in lengths if n < 60),  # 60자 미만 꼬리 청크 수
    }
    return collection, stats


# ── 3. 검색 (+ 리랭킹) ───────────────────────────────────────────────────────
def search_all(collection, questions: list[str], k: int):
    """질문마다 검색된 청크의 (doc_id, 텍스트) 목록을 순서대로 돌려준다."""
    k = min(k, collection.count())
    results = []
    for start in range(0, len(questions), 64):
        batch = questions[start:start + 64]
        res = collection.query(query_texts=batch, n_results=k, include=["documents", "metadatas"])
        for docs, metas in zip(res["documents"], res["metadatas"]):
            results.append([(m["doc_id"], d) for m, d in zip(metas, docs)])
    return results


def rerank_all(reranker, questions: list[str], all_hits):
    """이미 검색한 후보의 순서만 리랭커 점수로 다시 정한다. 후보 자체는 바뀌지 않는다."""
    reranked = []
    for q, hits in zip(questions, all_hits):
        scores = reranker.predict([[q, d] for _, d in hits])
        reranked.append([h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)])
    return reranked


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


# ── 5. 기록 ───────────────────────────────────────────────────────────────────
FIELDS = ["time", "embed", "chunker", "rerank", "n_docs", "chunk_size", "overlap",
          "n_chunks", "avg_len", "short_chunks", "hit@1", "hit@3", "hit@10", "mrr", "ms_per_q"]


def append_result(row: dict, path: Path = Path("results.csv")):
    """실험할 때마다 한 줄씩 쌓는다. 예전 형식의 파일이 있으면 이름을 바꿔 보관한다."""
    if path.exists():
        header = path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
        if header != FIELDS:
            old = path.with_name("results_old.csv")
            path.replace(old)
            print(f"(이전 형식의 결과 파일은 {old.name}로 보관)")
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_failures(name: str, questions: pd.DataFrame, all_hits, ranks):
    """상위 3개 안에 정답이 없던 질문을 모아 둔다. 왜 틀렸는지 읽어 보는 게 다음 개선의 출발점."""
    fails = [
        {"question": q, "answer_doc": d, "rank": r if r else "",
         "top1_doc": h[0][0] if h else "", "top1_text": h[0][1][:100] if h else ""}
        for q, d, h, r in zip(questions["question"], questions["doc_id"], all_hits, ranks)
        if r is None or r > 3
    ]
    pd.DataFrame(fails).to_csv(name, index=False, encoding="utf-8-sig")
    return len(fails)


def report(label: str, result: dict, ms_per_q: float):
    print(f"\n[{label}]")
    for name, value in result.items():
        print(f"  {name:7s} {value:.3f}")
    print(f"  질문당 시간 {ms_per_q:.1f} ms")


# ── 6. 실행 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunker", choices=["fixed", "sentence"], default="fixed")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--n-docs", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    print("데이터 불러오는 중...")
    docs, questions = load_korquad(args.n_docs)
    q_list = questions["question"].tolist()

    print(f"색인 만드는 중... (임베딩: {args.embed}, 청킹: {args.chunker})")
    t0 = time.perf_counter()
    collection, stats = build_index(docs, get_embed_fn(args.embed), args.chunker, args.chunk_size, args.overlap)
    print(f"문단 {len(docs)}개 → 청크 {stats['n_chunks']}개 (평균 {stats['avg_len']}자, "
          f"60자 미만 {stats['short_chunks']}개), 색인 {time.perf_counter() - t0:.0f}초")

    print("검색 중...")
    t0 = time.perf_counter()
    base_hits = search_all(collection, q_list, K)
    base_ms = (time.perf_counter() - t0) * 1000 / len(q_list)

    runs = [("리랭킹 없음", False, base_hits, base_ms)]
    if args.rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL)
        print("리랭킹 중... (같은 후보로 순서만 다시 매김)")
        t0 = time.perf_counter()
        rr_hits = rerank_all(reranker, q_list, base_hits)
        rr_ms = base_ms + (time.perf_counter() - t0) * 1000 / len(q_list)
        runs.append(("리랭킹 적용", True, rr_hits, rr_ms))

    short_embed = args.embed.split("/")[-1]
    for label, rerank, hits, ms in runs:
        ranks = [first_rank(h, d) for h, d in zip(hits, questions["doc_id"])]
        result = score(ranks)
        report(label, result, ms)
        append_result({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "embed": args.embed, "chunker": args.chunker, "rerank": rerank, "n_docs": len(docs),
            "chunk_size": args.chunk_size, "overlap": args.overlap, **stats,
            **{k: round(v, 3) for k, v in result.items()}, "ms_per_q": round(ms, 1),
        })
        name = f"failures_{short_embed}_{args.chunker}{'_rerank' if rerank else ''}.csv"
        n_fail = save_failures(name, questions, hits, ranks)
        print(f"  상위 3개 안에 정답이 없던 질문 {n_fail}개 → {name}")

    print("\nresults.csv에 기록했어요.")


if __name__ == "__main__":
    main()
