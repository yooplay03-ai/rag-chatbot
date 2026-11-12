"""
보안 테스트 — 열람 권한 필터와 민감정보 가리기.
권한 필터는 실제 ChromaDB(메모리)에 조항을 넣고 검색해 확인한다. 임베딩 모델 대신 글자 기반 가짜 벡터를 쓴다.
"""

import sys
from pathlib import Path

import chromadb
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chatbot   # noqa: E402
import security  # noqa: E402


class CharEmbedding(EmbeddingFunction):
    """'임금', '휴가' 같은 글자가 들어 있는지로 만든 아주 단순한 벡터 (테스트용)"""
    WORDS = ["임금", "지급", "월급", "휴가", "휴게", "징계"]

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [[1.0 + text.count(w) for w in self.WORDS] for text in input]

    @staticmethod
    def name() -> str:
        return "char-embedding-test"


@pytest.fixture
def collection(monkeypatch):
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("test_rules", embedding_function=CharEmbedding(),
                                          metadata={"hnsw:space": "cosine"})
    docs = [
        ("public_pay", "제52조(임금의 지급) 임금은 매월 25일에 지급한다.", "하늘테크 취업규칙"),
        ("public_leave", "제37조(하기휴가) 사원은 3일의 하기휴가를 사용할 수 있다.", "하늘테크 취업규칙"),
        ("hr_only", "인사 내부 지침: 임금 조정 대상자 명단과 지급 기준", "인사 내부 지침"),
        ("untagged", "징계 심의 회의록 (임금 감액 포함)", "징계 회의록"),
    ]
    security_map = {**security.ACCESS_BY_DOC, "인사 내부 지침": "인사팀"}
    monkeypatch.setattr(security, "ACCESS_BY_DOC", security_map)
    col.add(ids=[d[0] for d in docs], documents=[d[1] for d in docs],
            metadatas=[{"doc_id": d[2], "label": d[0], "access": security.access_for(d[2])} for d in docs])
    monkeypatch.setattr(chatbot, "get_collection", lambda: col)
    yield col
    client.delete_collection("test_rules")


def found(allowed):
    return {c["label"] for c in chatbot.search_candidates("임금 지급일이 언제야?", allowed)}


def test_untagged_document_is_denied_by_default():
    assert security.access_for("징계 회의록") == security.DENY


def test_employee_cannot_find_hr_only_document(collection):
    assert found(["전체", "개발팀"]) == {"public_pay", "public_leave"}


def test_hr_can_find_hr_only_document(collection):
    assert found(["전체", "인사팀"]) == {"public_pay", "public_leave", "hr_only"}


def test_untagged_document_is_found_by_nobody(collection):
    for dept in ["개발팀", "인사팀", "총무팀"]:
        assert "untagged" not in found(["전체", dept])


def test_default_search_is_public_only(collection):
    assert found(None) == {"public_pay", "public_leave"}


@pytest.mark.parametrize("text, expected", [
    ("제 번호 010-1234-5678인데 월급날이 언제야?", "제 번호 [전화번호]인데 월급날이 언제야?"),
    ("01012345678로 연락 주세요", "[전화번호]로 연락 주세요"),
    ("주민번호 900101-2345678 맞나요", "주민번호 [주민번호] 맞나요"),
    ("yuna@example.com으로 보내 줘", "[이메일]으로 보내 줘"),
    ("여름휴가는 며칠이야?", "여름휴가는 며칠이야?"),
])
def test_mask_pii(text, expected):
    assert security.mask_pii(text) == expected


def test_llm_receives_masked_question(monkeypatch):
    sent = []
    monkeypatch.setattr(chatbot, "search_candidates", lambda q, allowed=None: [{"text": "제52조 임금", "label": "제52조"}])
    monkeypatch.setattr(chatbot, "rerank", lambda q, c, top_k=3: [{**c[0], "rerank_score": 1.0}])
    monkeypatch.setattr(chatbot, "ask_groq", lambda prompt: sent.append(prompt) or "25일입니다.")
    chatbot.answer_question("제 번호 010-1234-5678인데 월급날이 언제야?")
    assert "010-1234-5678" not in sent[0]
    assert "[전화번호]" in sent[0]


# ── 검색 품질 보완 (도메인 평가셋에서 찾은 문제) ─────────────────────────────────
def test_expand_query_adds_rule_terms_only_when_needed():
    assert "연장근로" in chatbot.expand_query("야근하면 수당 더 줘요?")
    assert "상여금" in chatbot.expand_query("보너스 얼마야?")
    assert chatbot.expand_query("정년이 몇 살이야?") == "정년이 몇 살이야?"


def test_rerank_keeps_one_chunk_per_article(monkeypatch):
    class FakeReranker:
        def predict(self, pairs):
            return [0.9, 0.8, 0.7, 0.6]
    monkeypatch.setattr(chatbot, "get_reranker", lambda: FakeReranker())
    candidates = [{"label": "제35조", "text": "a"}, {"label": "제35조", "text": "b"},
                  {"label": "제34조", "text": "c"}, {"label": "제33조", "text": "d"}]
    assert [c["label"] for c in chatbot.rerank("1년 안 됐는데 휴가?", candidates)] == ["제35조", "제34조", "제33조"]


def test_prompt_asks_to_flag_terms_missing_from_documents():
    # 직접 쓴 질문 h07: '성과금'을 묻자 규정에 없는데도 상여금을 '성과금(상여금)'이라 부르며 답했다
    prompt = chatbot.build_prompt("성과금 얼마야?", [{"label": "제55조(상여금 지급)", "text": "제55조 ..."}])
    assert "문서에 없다고 먼저 밝혀" in prompt
    assert "어떤 뜻으로 이해했는지" in prompt
