"""
API 테스트 — 무거운 모델(bge-m3, 리랭커)과 Groq 호출은 가짜로 바꿔, API의 동작만 빠르게 확인한다.
실행: pip install pytest httpx  →  python -m pytest
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api       # noqa: E402
import chatbot   # noqa: E402
import security  # noqa: E402


class FakeCollection:
    def count(self):
        return 102


CALLS = []


def fake_answer(question, allowed=None):
    CALLS.append(allowed)
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
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-pytest-0123456789")
    CALLS.clear()
    monkeypatch.setattr(chatbot, "get_collection", lambda: FakeCollection())
    monkeypatch.setattr(chatbot, "get_reranker", lambda: None)
    monkeypatch.setattr(chatbot, "answer_question", fake_answer)
    with TestClient(api.app) as c:
        yield c


def login(client, username="employee", password="demo1234"):
    r = client.post("/login", json={"username": username, "password": password})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chunks": 102}


def test_ask_returns_answer_sources_and_timings(client):
    r = client.post("/ask", json={"question": "여름휴가는 며칠이야?"}, headers=login(client))
    assert r.status_code == 200
    body = r.json()
    assert "3일" in body["answer"]
    assert body["sources"][0]["label"] == "하늘테크 취업규칙 제37조(하기휴가)"
    assert set(body["sources"][0]) == {"label", "text", "rerank_score"}
    assert body["timings_ms"]["llm"] == 456.7


@pytest.mark.parametrize("question", ["", "   ", "가" * 501])
def test_invalid_question_is_rejected(client, question):
    r = client.post("/ask", json={"question": question}, headers=login(client))
    assert r.status_code == 422


def test_missing_api_key_returns_503(client, monkeypatch):
    def no_key(question, allowed=None):
        raise EnvironmentError("GROQ_API_KEY 없음")
    monkeypatch.setattr(chatbot, "answer_question", no_key)
    r = client.post("/ask", json={"question": "월급날이 언제야?"}, headers=login(client))
    assert r.status_code == 503


def test_llm_failure_returns_502(client, monkeypatch):
    def llm_down(question, allowed=None):
        raise RuntimeError("model_not_found")
    monkeypatch.setattr(chatbot, "answer_question", llm_down)
    r = client.post("/ask", json={"question": "월급날이 언제야?"}, headers=login(client))
    assert r.status_code == 502


def test_chat_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "하늘테크 사내 규정 챗봇" in r.text


# ── 로그인과 권한 ──────────────────────────────────────────────────────────────
def test_ask_without_token_is_401(client):
    r = client.post("/ask", json={"question": "월급날이 언제야?"})
    assert r.status_code == 401
    assert CALLS == []          # 검색·LLM까지 가지 않는다


def test_wrong_password_is_401(client):
    r = client.post("/login", json={"username": "employee", "password": "틀린비번"})
    assert r.status_code == 401


def test_tampered_token_is_401(client):
    token = login(client)["Authorization"].removeprefix("Bearer ")
    header, payload, signature = token.split(".")
    forged = security.jwt.encode({"sub": "employee", "dept": "인사팀"}, "attacker-guessed-secret-key-0123456789", algorithm="HS256")
    for bad in [f"{header}.{forged.split('.')[1]}.{signature}", forged]:   # 부서만 바꿔치기 / 다른 키로 서명
        r = client.post("/ask", json={"question": "월급날이 언제야?"}, headers={"Authorization": f"Bearer {bad}"})
        assert r.status_code == 401
    assert CALLS == []


def test_expired_token_is_401(client):
    old = security.create_token({"sub": "employee", "dept": "개발팀"}, minutes=-1)
    r = client.get("/me", headers={"Authorization": f"Bearer {old}"})
    assert r.status_code == 401


def test_permission_comes_from_token(client):
    client.post("/ask", json={"question": "월급날이 언제야?"}, headers=login(client, "employee"))
    client.post("/ask", json={"question": "월급날이 언제야?"}, headers=login(client, "hr"))
    assert CALLS == [["전체", "개발팀"], ["전체", "인사팀"]]


def test_me_returns_user_and_dept(client):
    r = client.get("/me", headers=login(client, "hr"))
    assert r.json() == {"user": "hr", "dept": "인사팀"}
