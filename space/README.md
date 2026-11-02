---
title: 사내 규정 RAG 챗봇
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: 가상 회사 취업규칙을 근거 조항과 함께 답하는 RAG 챗봇
---

# 사내 규정 RAG 챗봇 (데모)

가상 회사 "하늘테크 주식회사"의 취업규칙을 검색해, 근거 조항과 함께 답하는 RAG 챗봇입니다.
고용노동부 2026년 표준취업규칙을 바탕으로, 빈칸(근무 시간, 월급날, 여름휴가 등)을 가상 회사 값으로 채운 데모용 데이터입니다.

- 임베딩 검색: BAAI/bge-m3 + ChromaDB
- 리랭킹: BAAI/bge-reranker-v2-m3 (후보 7개 → 상위 3개)
- 답변 생성: Groq (openai/gpt-oss-120b)
- 서버: FastAPI (`/ask`, `/health`, `/docs`)

무료 CPU 서버라 답변까지 10초 이상 걸릴 수 있고, 한동안 사용이 없으면 잠들어 처음 열 때 몇 분 걸립니다.

코드와 검색 품질 평가, 속도 측정 과정: https://github.com/yooplay03-ai/rag-chatbot
