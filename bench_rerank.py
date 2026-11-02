"""
bench_rerank.py — 리랭킹을 빠르게 만드는 설정을 찾기 위한 측정 스크립트

API 응답을 재 보니 임베딩 검색 ~0.1초, 답변 생성 ~1초인데 리랭킹이 ~9초였다.
리랭커는 (질문, 조항) 쌍을 하나씩 끝까지 읽기 때문에,
  ① 읽을 후보 수(CANDIDATE_K)와  ② 쌍마다 읽는 길이(max_length, 토큰)
에 비례해 느려진다. 둘을 줄였을 때 속도와 결과가 어떻게 달라지는지 잰다.

기준(baseline) = 지금 설정(후보 10개, 길이 제한 없음)
- top1 일치   : 1등으로 뽑힌 청크가 기준과 같은 질문 수
- top3 겹침   : 상위 3개 청크 중 기준과 겹치는 비율 (LLM에 실제로 들어가는 근거)
  ※ 긴 조항은 청크 2개로 나뉘어 이름(label)이 같을 수 있어, 이름이 아니라 청크 본문으로 비교한다.

실행: python bench_rerank.py      (Groq 키 필요 없음, 5분 정도 걸림)
"""

import csv
import statistics
import time
import warnings

import chatbot

QUESTIONS = [
    "여름휴가는 며칠이야?",
    "월급날이 언제야?",
    "입사 1년 안 됐는데 연차가 있어?",
    "직장 내 괴롭힘을 당하면 어떻게 해야 해?",
    "점심시간은 몇 시부터야?",
    "출산휴가는 얼마나 쓸 수 있어?",
    "결혼하면 휴가 받을 수 있어?",
    "회사 그만두려면 언제까지 말해야 해?",
    "수습기간 동안 월급은 다 나와?",
    "징계에는 어떤 종류가 있어?",
    "야근하면 수당 받아?",
    "육아휴직은 언제 신청할 수 있어?",
]

# (후보 수, 최대 토큰 길이 — None은 제한 없음)
# 512토큰을 넘는 쌍이 없어 512는 '제한 없음'과 같으므로 뺐다.
CONFIGS = [(10, None), (10, 256), (7, None), (5, None), (5, 256)]


def set_max_length(model, value):
    # 최신 sentence-transformers는 이 값을 바꾸면 tokenizer.model_max_length도 같이 바뀐다.
    # 그래서 '제한 없음'은 처음 값(ORIGINAL_MAX)으로 되돌려야 한다. (1차 측정 때 이 버그로
    # 256 다음의 '제한 없음'이 실제로는 256으로 측정됐다.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.max_length = ORIGINAL_MAX if value is None else value


def main():
    import torch
    print(f"torch 스레드 수: {torch.get_num_threads()}")

    global ORIGINAL_MAX
    reranker = chatbot.get_reranker()
    ORIGINAL_MAX = reranker.tokenizer.model_max_length

    # 후보는 bge-m3 검색 결과라 매번 같다 → 한 번만 구해 두고, 리랭킹 시간만 잰다.
    cands = {q: chatbot.search_candidates(q) for q in QUESTIONS}

    tok_lens = [len(reranker.tokenizer(q, c["text"])["input_ids"])
                for q in QUESTIONS for c in cands[q]]
    print(f"(질문, 조항) 쌍의 토큰 수: 평균 {statistics.mean(tok_lens):.0f}, "
          f"최대 {max(tok_lens)}, 512 초과 {sum(t > 512 for t in tok_lens)}개, "
          f"256 초과 {sum(t > 256 for t in tok_lens)}개 / 전체 {len(tok_lens)}개")

    chatbot.rerank(QUESTIONS[0], cands[QUESTIONS[0]])   # 첫 실행은 느리므로 버린다(warm-up)

    baseline, rows = {}, []
    for k, max_len in CONFIGS:
        set_max_length(reranker, max_len)
        times, top1_same, top3_overlap, changed = [], 0, [], []
        for q in QUESTIONS:
            t0 = time.perf_counter()
            top = chatbot.rerank(q, cands[q][:k])
            times.append((time.perf_counter() - t0) * 1000)
            texts = [c["text"] for c in top]
            if (k, max_len) == CONFIGS[0]:
                baseline[q] = top
            base_texts = [c["text"] for c in baseline[q]]
            top1_same += texts[0] == base_texts[0]
            top3_overlap.append(len(set(texts) & set(base_texts)) / len(base_texts))
            if texts[0] != base_texts[0]:
                changed.append(f"  - {q}: 1등 {baseline[q][0]['label']} → {top[0]['label']}")
            elif set(texts) != set(base_texts):
                lost = [c["label"] for c in baseline[q] if c["text"] not in texts]
                changed.append(f"  - {q}: 3등 안에서 빠짐 {lost}")

        row = {
            "candidates": k,
            "max_length": max_len or "제한 없음",
            "rerank_ms_avg": round(statistics.mean(times)),
            "rerank_ms_median": round(statistics.median(times)),
            "top1_same": f"{top1_same}/{len(QUESTIONS)}",
            "top3_overlap": f"{statistics.mean(top3_overlap):.0%}",
        }
        rows.append(row)
        print(f"후보 {k:>2}개, 길이 {str(row['max_length']):>5} | "
              f"평균 {row['rerank_ms_avg']:>5}ms (중앙값 {row['rerank_ms_median']:>5}ms) | "
              f"top1 일치 {row['top1_same']}, top3 겹침 {row['top3_overlap']}")
        for line in changed:
            print(line)

    set_max_length(reranker, None)
    with open("rerank_bench.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("\n결과를 rerank_bench.csv에 저장했습니다.")


if __name__ == "__main__":
    main()
