"""
eval_answers.py — 답변 품질 평가 (도메인 평가셋)

eval_retrieval.py는 KorQuAD로 "검색"만 쟀다. KorQuAD 질문은 문단을 보고 만들어 단어가 많이 겹치므로
실제 직원 질문보다 쉽다. 그래서 하늘테크 취업규칙에 대해 직원 말투로 쓴 질문(data/domain_eval.json)으로
검색부터 최종 답변까지 잰다.

지표
  - 검색 Hit@3      : 정답 조항이 LLM에 넘기는 상위 3개 안에 들었는가
  - 근거 인용률      : 답변 끝 "근거"에 정답 조항을 밝혔는가
  - 사실 정확도      : 답에 꼭 있어야 할 숫자·표현(예: 25일)이 들어 있는가
  - 거절률          : 규정에 없는 질문(재택근무, 반차 등)에 지어내지 않고 모른다고 답했는가

실행:
  python eval_answers.py --retrieval-only      # LLM 없이 검색만 (API 키 불필요)
  python eval_answers.py                       # 전체. GROQ_API_KEY 필요
  python eval_answers.py --sleep 2             # 무료 API 요청 한도를 피하려고 질문 사이에 2초 쉬기
  python eval_answers.py --set holdout         # 개선 뒤에 새로 쓴 질문으로 확인 (과대평가 방지)
  python eval_answers.py --set holdout --ids h03,h07   # 일부 질문만 다시 확인
결과: 화면 표 + eval_answers.csv (질문별 결과)
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

EVAL_PATH = Path(__file__).parent / "data" / "domain_eval.json"
HOLDOUT_PATH = Path(__file__).parent / "data" / "holdout_eval.json"   # 개선 후에 처음 보는 질문
OUT_CSV = Path(__file__).parent / "eval_answers.csv"
ARTICLE_RE = re.compile(r"제\d+조")
# "모른다"는 답의 표현들. 규칙 기반이라 완벽하지 않으므로 CSV에서 실제 답을 함께 확인한다.
REFUSAL_WORDS = ["모르", "알 수 없", "없습니다", "없어요", "포함되어 있지 않", "명시되어 있지 않",
                 "나와 있지 않", "규정되어 있지 않", "찾을 수 없", "확인되지 않", "언급되어 있지 않", "정해져 있지 않"]


# ── 채점 (모델 없이 테스트할 수 있게 순수 함수로 둔다) ────────────────────────────
def articles_in(text: str) -> set[str]:
    """'하늘테크 취업규칙 제52조(임금...)' → {'제52조'}"""
    return set(ARTICLE_RE.findall(text or ""))


def hit_at_k(source_labels: list[str], expected: list[str], k: int = 3) -> bool:
    found = set()
    for label in source_labels[:k]:
        found |= articles_in(label)
    return bool(found & set(expected))


def cites_expected(answer: str, expected: list[str], source_labels: list[str]) -> bool:
    """LLM이 실제로 받은 조항 중 정답 조항을 답에 밝혔는가.
    받지 않은 조항 번호는 인정하지 않는다. 다른 조항 본문의 "제29조에 따라" 같은 문구를 옮겨 적은 것일 수 있어서다
    (첫 평가에서 q21이 이렇게 잘못 '인용 O'로 채점됐다)."""
    seen = set()
    for label in source_labels:
        seen |= articles_in(label)
    return bool(articles_in(answer) & set(expected) & seen)


def has_facts(answer: str, must: list[str]) -> bool:
    """must의 각 항목이 답에 있는가. '12:00|12시'처럼 |로 같은 뜻의 다른 표기를 허용한다
    (개선 후 평가에서 q04가 "12 시부터 13 시까지"라고 맞게 답했는데 '12:00'만 찾아 틀린 것으로 채점됐다)."""
    squeezed = re.sub(r"\s+", "", answer or "")
    return all(any(re.sub(r"\s+", "", alt) in squeezed for alt in m.split("|")) for m in must)


def is_refusal(answer: str) -> bool:
    return any(w in (answer or "") for w in REFUSAL_WORDS)


def score(item: dict, source_labels: list[str], answer: str | None) -> dict:
    """한 질문의 채점 결과. answer가 None이면 검색만 채점한다."""
    if item.get("answerable", True) is False:
        return {"refused": None if answer is None else is_refusal(answer)}
    result = {"hit3": hit_at_k(source_labels, item["expected"])}
    if answer is not None:
        result["cited"] = cites_expected(answer, item["expected"], source_labels)
        result["facts"] = has_facts(answer, item.get("must", []))
        if item.get("also_unanswerable"):          # 예: "점심시간은 언제고 식비 줘?" → 식비는 규정에 없음
            result["refused"] = is_refusal(answer)
    return result


def summarize(rows: list[dict]) -> dict:
    def rate(key, subset):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return (sum(vals), len(vals))
    answerable = [r for r in rows if r["answerable"]]
    return {
        "검색 Hit@3": rate("hit3", answerable),
        "근거 인용률": rate("cited", answerable),
        "사실 정확도": rate("facts", answerable),
        "거절률 (규정에 없는 질문·부분)": rate("refused", rows),
    }


# ── 무료 API 요청 한도 대응 ────────────────────────────────────────────────────
MAX_WAIT_SECONDS = 15 * 60


def retry_after(message: str) -> float | None:
    """Groq 429 오류 메시지의 'try again in 4m23.952s'에서 기다릴 초를 꺼낸다. 한도 오류가 아니면 None."""
    if "429" not in message and "rate limit" not in message.lower():
        return None
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    if not m:
        return 60.0
    return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1


def answer_with_retry(answer_fn, question: str, retries: int = 5) -> dict:
    """한도에 걸리면 알려 준 시간만큼 기다렸다가 다시 묻는다. 평가 도중 멈춰 결과를 잃지 않게 하려는 것."""
    for attempt in range(retries):
        try:
            return answer_fn(question)
        except Exception as e:  # groq.RateLimitError 등 (groq를 import하지 않고도 테스트할 수 있게)
            wait = retry_after(str(e))
            if wait is None or wait > MAX_WAIT_SECONDS or attempt == retries - 1:
                raise
            print(f"   요청 한도에 걸려 {wait:.0f}초 기다렸다가 다시 시도합니다 ({attempt + 1}/{retries})")
            time.sleep(wait)


# ── 실행 ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="LLM 없이 검색만 평가")
    ap.add_argument("--sleep", type=float, default=0.0, help="질문 사이 쉬는 시간(초)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", choices=["domain", "holdout"], default="domain")
    ap.add_argument("--ids", default="", help="쉼표로 구분한 질문 id만 실행 (예: h03,h07)")
    args = ap.parse_args()

    import chatbot   # 모델 로딩이 무거워서 실행할 때만 불러온다

    path = HOLDOUT_PATH if args.set == "holdout" else EVAL_PATH
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    out_csv = OUT_CSV.with_name(f"eval_answers_{args.set}.csv")
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i["id"] in wanted]
        out_csv = out_csv.with_name(f"eval_answers_{args.set}_selected.csv")   # 전체 결과를 덮어쓰지 않게
    if args.limit:
        items = items[: args.limit]

    rows = []
    for i, item in enumerate(items, 1):
        answerable = item.get("answerable", True)
        if args.retrieval_only:
            sources = chatbot.retrieve(item["question"]) if answerable else []
            answer = None
        else:
            result = answer_with_retry(chatbot.answer_question, item["question"])
            sources, answer = result["sources"], result["answer"]
        labels = [s["label"] for s in sources]
        sc = score(item, labels, answer)
        rows.append({
            "id": item["id"], "question": item["question"], "answerable": answerable,
            "expected": ",".join(item.get("expected", [])), "top3": " | ".join(labels[:3]),
            "hit3": sc.get("hit3"), "cited": sc.get("cited"), "facts": sc.get("facts"),
            "refused": sc.get("refused"), "answer": (answer or "").replace("\n", " "),
        })
        mark = {True: "O", False: "X", None: "-"}
        print(f"[{i:2d}/{len(items)}] {item['id']} 검색 {mark[sc.get('hit3')]} "
              f"인용 {mark[sc.get('cited')]} 사실 {mark[sc.get('facts')]} 거절 {mark[sc.get('refused')]}  {item['question']}")
        if args.sleep and not args.retrieval_only:
            time.sleep(args.sleep)

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:   # 엑셀에서 한글이 안 깨지게 utf-8-sig
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n| 지표 | 결과 |\n|---|---|")
    for name, (ok, n) in summarize(rows).items():
        print(f"| {name} | {ok}/{n} ({ok / n:.1%}) |" if n else f"| {name} | - |")
    print(f"\n질문별 결과와 실제 답변: {out_csv.name}")


if __name__ == "__main__":
    main()
