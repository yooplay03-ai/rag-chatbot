"""
API 테스트 — 무거운 모델(bge-m3, 리랭커)과 Groq 호출은 가짜로 바꿔, API의 동작만 빠르게 확인한다.
실행: pip install pytest httpx  →  python -m pytest
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api      # noqa: E402
import chatbot  # noqa: E402


class FakeCollection:
    def count(self):
        return 102


def fake_answer(question):
    return {
        "answer": "여름휴가는 3일입니다.\n근거: 하늘테크 취업규칙 제37조(하기휴가)",
        "sources": [{
            "label": "하늘테크 취업규칙 제37조(하기휴가)",
            "text": "제37조(하기휴가) 사원은 7월 15일부터 8월 31일까지 사이에 3일의 하기휴가를 사용할 수 있다.",
            "rerank_score": 0.3166,
            "doc_id": "하늘테크 취업규칙",   # 응답 모델에 없는 값은 API가 걸러서 내보내지 않는다
            "distance": 0.21,
        }],
        "timings_ms": {"search": 12.3, "rerank": 45.6, "llm": 456.7},
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(chatbot, "get_collection", lambda: FakeCollection())
    monkeypatch.setattr(chatbot, "get_reranker", lambda: None)
    monkeypatch.setattr(chatbot, "answer_question", fake_answer)
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chunks": 102}


def test_ask_returns_answer_sources_and_timings(client):
    r = client.post("/ask", json={"question": "여름휴가는 며칠이야?"})
    assert r.status_code == 200
    body = r.json()
    assert "3일" in body["answer"]
    assert body["sources"][0]["label"] == "하늘테크 취업규칙 제37조(하기휴가)"
    assert set(body["sources"][0]) == {"label", "text", "rerank_score"}
    assert body["timings_ms"]["llm"] == 456.7


@pytest.mark.parametrize("question", ["", "   ", "가" * 501])
def test_invalid_question_is_rejected(client, question):
    r = client.post("/ask", json={"question": question})
    assert r.status_code == 422


def test_missing_api_key_returns_503(client, monkeypatch):
    def no_key(question):
        raise EnvironmentError("GROQ_API_KEY 없음")
    monkeypatch.setattr(chatbot, "answer_question", no_key)
    r = client.post("/ask", json={"question": "월급날이 언제야?"})
    assert r.status_code == 503


def test_llm_failure_returns_502(client, monkeypatch):
    def llm_down(question):
        raise RuntimeError("model_not_found")
    monkeypatch.setattr(chatbot, "answer_question", llm_down)
    r = client.post("/ask", json={"question": "월급날이 언제야?"})
    assert r.status_code == 502


def test_chat_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "하늘테크 사내 규정 챗봇" in r.text
