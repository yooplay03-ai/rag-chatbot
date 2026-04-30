# RAG 챗봇 (ChromaDB + Groq + Reranking)

> 파이썬으로 만든 RAG(Retrieval-Augmented Generation) 챗봇입니다.
> ChromaDB에 문서를 저장하고, Reranking으로 검색 품질을 높여 Groq LLaMA가 답변합니다.

---

## 프로젝트 소개

AI 전공자로서 RAG 시스템의 데이터 효율성을 직접 실험해보기 위해 만든 프로젝트입니다.
문서를 청킹(chunking)해서 벡터 DB에 저장하고, Reranking으로 검색 정확도를 높인 뒤
LLM에 전달하는 전체 파이프라인을 구현했습니다.

```
사용자 질문
  → 임베딩 변환
  → ChromaDB에서 후보 10개 검색
  → Reranker(bge-reranker-v2-m3)로 재정렬 → 상위 3개 선별
  → 프롬프트 조합 (컨텍스트 + 질문)
  → Groq LLaMA API 호출
  → 답변 출력
```

---

## 파일 구조

```
rag_chatbot/
├── ingest.py        # 문서 추가 / 삭제 / 조회
├── chatbot.py       # RAG 파이프라인 + 대화형 CLI
├── README.md        # 프로젝트 설명
└── chroma_db/       # ChromaDB 로컬 저장소 (자동 생성)
```

---

## 설치

```bash
pip install chromadb sentence-transformers groq
```

---

## 사용 방법

### 1단계 — Groq API 키 설정

[https://console.groq.com](https://console.groq.com) 에서 무료 가입 후 API 키 발급

```bash
# Windows
set GROQ_API_KEY=gsk_여기에키입력

# Mac/Linux
export GROQ_API_KEY=gsk_여기에키입력
```

### 2단계 — 문서 추가

```bash
python ingest.py
```

### 3단계 — 챗봇 실행

```bash
python chatbot.py
```

```
==================================================
  RAG 챗봇 (ChromaDB + Groq)
  종료: 'q' 또는 'quit' 입력
==================================================

질문: RAG가 뭐야?

── 검색된 청크 (reranking 적용) ──
  1. [rag_intro] rerank 점수: 0.9821
     RAG(Retrieval-Augmented Generation)는 검색 기반 생성 방식으로 ...

답변:
RAG는 Retrieval-Augmented Generation의 약자로, 외부 문서를 검색해
LLM의 답변 품질을 높이는 기법입니다 ...
```

### 문서 관리

```python
from ingest import list_documents, delete_document, delete_all

list_documents()            # 저장된 문서 목록 확인
delete_document("문서이름")  # 특정 문서 삭제
delete_all()                # 전체 삭제 (주의)
```

---

## 커스터마이징

| 파일 | 변수 | 설명 | 기본값 |
|------|------|------|--------|
| `ingest.py` | `CHUNK_SIZE` | 청크 크기 | 500자 |
| `ingest.py` | `CHUNK_OVERLAP` | 청크 겹침 | 50자 |
| `ingest.py` | `EMBED_MODEL` | 임베딩 모델 | all-MiniLM-L6-v2 |
| `chatbot.py` | `CANDIDATE_K` | 임베딩 검색 후보 수 | 10 |
| `chatbot.py` | `TOP_K` | LLM에 전달할 최종 청크 수 | 3 |
| `chatbot.py` | `RERANK_MODEL` | 리랭킹 모델 | bge-reranker-v2-m3 |
| `chatbot.py` | `GROQ_MODEL` | LLM 모델 | llama-3.3-70b-versatile |

---

## 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 벡터 DB | ChromaDB |
| 임베딩 모델 | SentenceTransformers (all-MiniLM-L6-v2) |
| 리랭킹 모델 | BAAI/bge-reranker-v2-m3 |
| LLM | Groq (LLaMA 3.3 70B) |
| 언어 | Python 3.10+ |

---

## 개선 가능한 점

- 임베딩 모델을 `BAAI/bge-m3`으로 교체 → 한국어 검색 품질 향상
- `RecursiveCharacterTextSplitter` 도입 → 문장 경계 존중 청킹
- PDF, DOCX 등 다양한 파일 형식 지원
- 대화 히스토리 유지 (멀티턴 대화)
- 웹 UI 추가 (Streamlit 등)

---

## 라이선스

MIT License
