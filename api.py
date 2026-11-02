"""
api.py — 사내 규정 RAG 챗봇을 FastAPI 서버로 제공한다.

실행:  uvicorn api:app --reload        (내 컴퓨터, http://127.0.0.1:8000)
배포:  Dockerfile 참고 (Hugging Face Spaces, 포트 7860)

엔드포인트
- GET  /        : 브라우저용 채팅 화면 (static/index.html)
- GET  /health  : 서버 상태와 저장된 청크 수
- POST /ask     : {"question": "..."} → 답변, 근거 조항, 단계별 걸린 시간
- GET  /docs    : FastAPI가 자동으로 만들어 주는 API 문서와 테스트 화면
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import chatbot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-api")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 켜질 때 무거운 모델과 DB를 한 번만 불러 둔다. 첫 질문도 빠르게 답하기 위해서다.
    collection = chatbot.get_collection()
    chatbot.get_reranker()
    log.info("준비 완료: 저장된 청크 %d개", collection.count())
    yield


app = FastAPI(title="사내 규정 RAG 챗봇 API", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, examples=["여름휴가는 며칠이야?"])


class Source(BaseModel):
    label: str
    text: str
    rerank_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    timings_ms: dict[str, float]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": chatbot.get_collection().count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="질문을 입력해 주세요.")
    try:
        result = chatbot.answer_question(question)
    except EnvironmentError as e:            # API 키가 없을 때
        log.error("설정 오류: %s", e)
        raise HTTPException(status_code=503, detail="서버에 LLM API 키가 설정되지 않았습니다.")
    except Exception as e:                   # LLM 호출 실패 (모델 지원 종료, 한도 초과 등)
        log.exception("답변 생성 실패")
        raise HTTPException(status_code=502, detail=f"답변 생성 중 오류가 발생했습니다: {type(e).__name__}")

    t = result["timings_ms"]
    log.info("질문 %d자 | 근거 %s | 검색 %.0fms, 리랭킹 %.0fms, LLM %.0fms",
             len(question), [s["label"] for s in result["sources"]],
             t["search"], t["rerank"], t["llm"])
    return result
