# 🤖 RAG 챗봇 (ChromaDB + Gemini)

> 파이썬으로 만든 간단한 RAG(Retrieval-Augmented Generation) 챗봇입니다.  
> ChromaDB에 문서를 저장하고, 질문과 유사한 내용을 검색해 Gemini가 답변합니다.

---

## 📌 프로젝트 소개

AI 전공자로서 RAG 시스템의 데이터 효율성을 직접 실험해보기 위해 만든 프로젝트입니다.  
문서를 청킹(chunking)해서 벡터 DB에 저장하고, 검색된 컨텍스트를 LLM에 전달하는 전체 파이프라인을 구현했습니다.

```
사용자 질문
  → 임베딩 변환
  → ChromaDB에서 유사 청크 검색 (top-k)
  → 프롬프트 조합 (컨텍스트 + 질문)
  → Gemini API 호출
  → 답변 출력
```

---

## 🗂️ 파일 구조

```
rag_chatbot/
├── ingest.py        # 문서 추가 / 삭제 / 조회
├── chatbot.py       # RAG 파이프라인 + 대화형 CLI
├── requirements.txt # 패키지 목록
└── chroma_db/       # ChromaDB 로컬 저장소 (자동 생성)
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

또는 직접 설치:

```bash
pip install chromadb sentence-transformers google-genai
```

---

## 🚀 사용 방법

### 1단계 — 문서 추가

```python
from ingest import add_document

add_document("저장할 텍스트 내용...", doc_id="문서이름", metadata={"topic": "AI"})
```

또는 샘플 데이터로 바로 테스트:

```bash
python ingest.py
```

### 2단계 — 챗봇 실행

```bash
python chatbot.py
```

```
==================================================
  RAG 챗봇 (ChromaDB + Gemini)
  종료: 'q' 또는 'quit' 입력
==================================================

질문: RAG가 뭐야?

── 검색된 청크 ──
  1. [rag_intro] 유사도: 28.78%
     RAG(Retrieval-Augmented Generation)는 검색 기반 생성 방식으로 ...

답변:
RAG는 Retrieval-Augmented Generation의 약자로, 외부 문서를 검색해
LLM의 답변 품질을 높이는 기법입니다 ...
```

### 문서 관리

```python
from ingest import list_documents, delete_document, delete_all

list_documents()           # 저장된 문서 목록 확인
delete_document("문서이름") # 특정 문서 삭제
delete_all()               # 전체 삭제 (주의)
```

---

## 🛠️ 커스터마이징

| 파일 | 변수 | 설명 | 기본값 |
|------|------|------|--------|
| `ingest.py` | `CHUNK_SIZE` | 청크 크기 | 500자 |
| `ingest.py` | `CHUNK_OVERLAP` | 청크 겹침 | 50자 |
| `ingest.py` | `EMBED_MODEL` | 임베딩 모델 | all-MiniLM-L6-v2 |
| `chatbot.py` | `TOP_K` | 검색 청크 수 | 3 |
| `chatbot.py` | `GEMINI_MODEL` | Gemini 모델 | gemini-2.0-flash-lite |

---

## 📝 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 벡터 DB | ChromaDB |
| 임베딩 모델 | SentenceTransformers (all-MiniLM-L6-v2) |
| LLM | Google Gemini API |
| 언어 | Python 3.10+ |

---

## 📖 관련 글

- [AI 전공자로서 RAG 시스템의 데이터 효율성을 고민하다](https://velog.io) — Velog 기술 블로그

---

## 📄 라이선스

MIT License
