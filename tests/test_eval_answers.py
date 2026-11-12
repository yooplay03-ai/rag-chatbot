"""답변 평가의 채점 규칙 테스트 (모델·API 키 없이)"""

import json
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_answers as ev  # noqa: E402


def test_article_numbers_do_not_partially_match():
    assert ev.articles_in("하늘테크 취업규칙 제52조(임금의 계산 및 지급방법)") == {"제52조"}
    assert not ev.hit_at_k(["하늘테크 취업규칙 제5조(채용)"], ["제52조"])


def test_hit_at_3_only_counts_top_three():
    labels = ["… 제24조(휴게)", "… 제37조(하기휴가)", "… 제23조(근로시간)", "… 제52조(임금)"]
    assert not ev.hit_at_k(labels, ["제52조"])
    assert ev.hit_at_k(labels, ["제23조"])


def test_answerable_scoring():
    item = {"id": "q01", "question": "월급 언제?", "expected": ["제52조"], "must": ["25일"]}
    good = "월급은 매월 25 일에 지급됩니다.\n근거: 하늘테크 취업규칙 제52조(임금의 계산 및 지급방법)"
    assert ev.score(item, ["… 제52조(임금)"], good) == {"hit3": True, "cited": True, "facts": True}
    wrong_cite = "월급은 매월 25일입니다. 근거: 제51조"
    assert ev.score(item, ["… 제52조(임금)"], wrong_cite)["cited"] is False
    wrong_fact = "월급은 매월 10일입니다. 근거: 제52조"
    assert ev.score(item, ["… 제52조(임금)"], wrong_fact)["facts"] is False


def test_unanswerable_scoring():
    item = {"id": "n04", "question": "반차 쓸 수 있어?", "answerable": False}
    assert ev.score(item, [], "제공된 규정에는 반차에 관한 내용이 없습니다.")["refused"] is True
    assert ev.score(item, [], "네, 연차를 반으로 나눠 반차로 쓸 수 있습니다.")["refused"] is False


def test_retrieval_only_leaves_answer_metrics_empty():
    item = {"id": "q01", "question": "월급 언제?", "expected": ["제52조"]}
    assert ev.score(item, ["… 제52조(임금)"], None) == {"hit3": True}


def test_summary_counts_by_group():
    rows = [
        {"answerable": True, "hit3": True, "cited": True, "facts": False, "refused": None},
        {"answerable": True, "hit3": False, "cited": False, "facts": True, "refused": None},
        {"answerable": False, "hit3": None, "cited": None, "facts": None, "refused": True},
    ]
    s = ev.summarize(rows)
    assert s["검색 Hit@3"] == (1, 2) and s["거절률 (규정에 없는 질문·부분)"] == (1, 1)


@pytest.mark.parametrize("path", [ev.EVAL_PATH, ev.HOLDOUT_PATH])
def test_eval_set_is_well_formed(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = json.loads((path.parent / "haneul_rules.json").read_text(encoding="utf-8"))
    known = {a["article"] for a in rules["articles"]}
    ids = [i["id"] for i in data["items"]]
    assert len(ids) == len(set(ids))
    for item in data["items"]:
        if item.get("answerable", True):
            assert item["expected"] and set(item["expected"]) <= known, item["id"]


def test_partly_unanswerable_question_scores_both_parts():
    item = {"id": "h04", "question": "점심시간은 언제고 식비 줘?", "expected": ["제24조"],
            "must": ["12:00"], "also_unanswerable": True}
    answer = "점심시간은 12:00부터 13:00까지입니다. 식비 지원은 규정에 없습니다. 근거: 제24조(휴게)"
    assert ev.score(item, ["… 제24조(휴게)"], answer) == {"hit3": True, "cited": True, "facts": True, "refused": True}


def test_citing_an_article_that_was_not_retrieved_does_not_count():
    # 첫 평가 q21: 제29조를 검색하지 못했는데, 제51조 본문의 "제29조에 따라"를 옮겨 적어 인용으로 잡혔다
    item = {"id": "q21", "question": "야근하면 수당 더 줘요?", "expected": ["제29조"], "must": []}
    labels = ["… 제51조(임금의 구성항목)", "… 제54조(휴업수당)", "… 제31조(적용제외)"]
    answer = "제51조에 따르면 야간근로는 제29조에 따라 가산합니다. 근거: 제51조"
    assert ev.score(item, labels, answer)["cited"] is False


def test_facts_accept_alternative_spellings():
    # 개선 후 평가 q04: "12 시부터 13 시까지"는 맞는 답인데 '12:00'만 찾아 틀린 것으로 채점됐다
    must = ["12:00|12시", "13:00|13시"]
    assert ev.has_facts("점심시간은 12 시부터 13 시까지입니다.", must)
    assert ev.has_facts("12:00부터 13:00까지", must)
    assert not ev.has_facts("11시부터 12시까지", must)


def test_retry_after_reads_wait_time_from_rate_limit_error():
    msg = "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` ... Please try again in 4m23.952s."
    assert ev.retry_after(msg) == pytest.approx(4 * 60 + 23.952 + 1)
    assert ev.retry_after("Error code: 429 - Please try again in 7.5s") == pytest.approx(8.5)
    assert ev.retry_after("Error code: 401 - invalid api key") is None


def test_answer_with_retry_waits_then_succeeds(monkeypatch):
    monkeypatch.setattr(ev.time, "sleep", lambda s: None)
    calls = []

    def flaky(q):
        calls.append(q)
        if len(calls) == 1:
            raise RuntimeError("Error code: 429 - Please try again in 2s")
        return {"answer": "ok", "sources": []}
    assert ev.answer_with_retry(flaky, "월급 언제?")["answer"] == "ok"
    assert len(calls) == 2
