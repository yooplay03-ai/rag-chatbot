"""
api.py — 사내 규정 RAG 챗봇을 FastAPI 서버로 제공한다.

실행:  uvicorn api:app --reload        (내 컴퓨터, http://127.0.0.1:8000)
배포:  Dockerfile 참고 (Hugging Face Spaces, 포트 7860)

엔드포인트
- GET  /        : 브라우저용 채팅 화면 (static/index.html)
- GET  /health  : 서버 상태와 저장된 청크 수
- POST /login   : {"username", "password"} → JWT (30분 유효)
- GET  /me      : 토큰의 사용자와 부서
- POST /ask     : {"question": "..."} → 답변, 근거 조항, 단계별 걸린 시간 (Authorization: Bearer 토큰 필요)
- GET  /docs    : FastAPI가 자동으로 만들어 주는 API 문서와 테스트 화면 (오른쪽 위 Authorize에 토큰 입력)

필요한 환경변수: GROQ_API_KEY(답변 생성), JWT_SECRET(토큰 서명)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

import chatbot
import security

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


bearer = HTTPBearer(auto_error=False)


def current_user(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    """요청 헤더의 토큰을 검사해 사용자를 돌려준다. 없거나, 위조됐거나, 만료됐으면 401."""
    if cred is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    try:
        return security.decode_token(cred.credentials)
    except EnvironmentError:
        raise HTTPException(status_code=503, detail="서버에 JWT_SECRET이 설정되지 않았습니다.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="토큰이 유효하지 않거나 만료되었습니다. 다시 로그인해 주세요.")


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=100)


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


@app.post("/login")
def login(req: LoginRequest):
    user = security.authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    try:
        token = security.create_token(user)
    except EnvironmentError:
        raise HTTPException(status_code=503, detail="서버에 JWT_SECRET이 설정되지 않았습니다.")
    log.info("로그인: %s (%s)", user["sub"], user["dept"])
    return {"token": token, "user": user["sub"], "dept": user["dept"]}


@app.get("/me")
def me(user: dict = Depends(current_user)):
    return {"user": user["sub"], "dept": user["dept"]}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, user: dict = Depends(current_user)):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="질문을 입력해 주세요.")
    try:
        # 권한은 요청 본문이 아니라 서명된 토큰에서 꺼낸다. 사용자가 부서를 바꿔 보낼 수 없다.
        result = chatbot.answer_question(question, allowed=security.allowed_for(user))
    except EnvironmentError as e:            # API 키가 없을 때
        log.error("설정 오류: %s", e)
        raise HTTPException(status_code=503, detail="서버에 LLM API 키가 설정되지 않았습니다.")
    except Exception as e:                   # LLM 호출 실패 (모델 지원 종료, 한도 초과 등)
        log.exception("답변 생성 실패")
        raise HTTPException(status_code=502, detail=f"답변 생성 중 오류가 발생했습니다: {type(e).__name__}")

    t = result["timings_ms"]
    log.info("%s(%s) 질문 %d자 | 근거 %s | 검색 %.0fms, 리랭킹 %.0fms, LLM %.0fms",
             user["sub"], user["dept"], len(question), [s["label"] for s in result["sources"]],
             t["search"], t["rerank"], t["llm"])
    return result
