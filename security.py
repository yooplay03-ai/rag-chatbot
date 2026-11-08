"""
security.py — 로그인(JWT), 문서 열람 권한, 민감정보 가리기

데이터가 밖으로 나가는 곳은 LLM 호출 한 곳(chatbot.ask_groq)뿐이다. 그래서 세 가지를 둔다.
1. 로그인: 직원만 질문할 수 있게 JWT를 발급하고, 요청마다 서명과 만료를 검사한다.
2. 열람 권한: 문서마다 볼 수 있는 부서를 메타데이터로 붙이고, 검색 단계에서 걸러 낸다.
   LLM에게 "보여 주지 마"라고 지시하는 대신 애초에 넘기지 않는 방식이다.
3. 민감정보 가리기: 질문 속 주민번호·전화번호·이메일을 LLM으로 보내기 전에 가린다.
"""

import hashlib
import hmac
import os
import re
from datetime import datetime, timedelta, timezone

import jwt

# ── 문서 열람 권한 ─────────────────────────────────────────────────────────────
# 취업규칙은 근로기준법 제14조에 따라 근로자가 자유롭게 열람할 수 있어야 하므로 전 직원 공개다.
# 인사 내부 지침처럼 일부 부서만 봐야 하는 문서를 넣을 때 여기에 부서를 적는다.
PUBLIC = "전체"
DENY = "비공개"
ACCESS_BY_DOC = {
    "하늘테크 취업규칙": PUBLIC,
}


def access_for(doc_id: str) -> str:
    """문서의 열람 권한. 등록하지 않은 문서는 아무도 못 보게 막는다(기본 거부)."""
    return ACCESS_BY_DOC.get(doc_id, DENY)


def allowed_for(user: dict) -> list[str]:
    """사용자가 볼 수 있는 권한 꼬리표 목록: 전 직원 공개 + 내 부서."""
    return [PUBLIC, user["dept"]]


# ── 민감정보 가리기 ─────────────────────────────────────────────────────────────
PII_PATTERNS = [
    (re.compile(r"\d{6}-?[1-4]\d{6}"), "[주민번호]"),
    (re.compile(r"01[016789]-?\d{3,4}-?\d{4}"), "[전화번호]"),
    # \w는 한글도 글자로 쳐서 "a@b.com으로"의 "으로"까지 먹으므로 영문·숫자만 허용
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"), "[이메일]"),
]


def mask_pii(text: str) -> str:
    for pattern, label in PII_PATTERNS:
        text = pattern.sub(label, text)
    return text


# ── 데모 계정 ──────────────────────────────────────────────────────────────────
# 비밀번호는 원문 대신 PBKDF2 해시로 둔다. 실제 서비스라면 사내 계정 시스템(SSO)과 연결한다.
PBKDF2_ITERATIONS = 310_000


def hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS)
    return digest.hex()


DEMO_USERS = {
    # 아이디: 부서, 솔트, 해시  (데모 비밀번호는 README에 적어 둠)
    "employee": {"dept": "개발팀", "salt": "haneul-employee",
                 "hash": "4e560992215a2e9e48c7ac483583f43895db3bda23f25cc20ef0075c1b4e6a06"},
    "hr": {"dept": "인사팀", "salt": "haneul-hr",
           "hash": "99431628052f8123a886e8b8e496b0f3c2227b628003fa2c8fb622d8a21ed1c0"},
}


def authenticate(username: str, password: str) -> dict | None:
    user = DEMO_USERS.get(username)
    if user is None:
        return None
    if not hmac.compare_digest(hash_password(password, user["salt"]), user["hash"]):
        return None
    return {"sub": username, "dept": user["dept"]}


# ── JWT ────────────────────────────────────────────────────────────────────────
ALGORITHM = "HS256"
TOKEN_MINUTES = 30


def get_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise EnvironmentError("JWT_SECRET 환경변수가 설정되지 않았습니다.")
    return secret


def create_token(user: dict, minutes: int = TOKEN_MINUTES) -> str:
    payload = {**user, "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)}
    return jwt.encode(payload, get_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """서명이 틀리거나(위조) 만료됐으면 jwt.InvalidTokenError를 낸다."""
    return jwt.decode(token, get_secret(), algorithms=[ALGORITHM])
