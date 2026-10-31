# RAG 챗봇 (ChromaDB + Groq + Reranking)

> 파이썬으로 만든 RAG(Retrieval-Augmented Generation) 챗봇입니다.
> ChromaDB에 문서를 저장하고, Reranking으로 검색 품질을 높여 Groq의 LLM이 답변합니다.
> KorQuAD 기반 평가셋으로 검색 품질을 측정하며 개선하고 있습니다.

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
  → Groq API 호출 (openai/gpt-oss-120b)
  → 답변 출력
```

---

## 검색 품질 평가

RAG에서 답이 틀리면 원인이 검색인지 생성인지부터 나눠 봐야 하므로, LLM을 빼고 **검색 단계만** 따로 평가했습니다.

### 실험 1: 임베딩 모델과 리랭킹 (문단 300개)

**방법**
- 질문마다 정답 문단이 정해져 있는 [KorQuAD 1.0](https://huggingface.co/datasets/KorQuAD/squad_kor_v1) 검증 데이터에서 문단 300개와 질문 300개(문단당 1개)를 뽑아 사용
- 챗봇과 같은 청킹 함수(500자, 겹침 50자)로 문단을 잘라 ChromaDB에 저장
- 지표
  - **Hit@3**: 정답 문단이 상위 3개 안에 든 비율. 챗봇이 상위 3개 청크만 LLM에 넘기므로 핵심 지표
  - **Hit@10**: 후보 10개 안에 든 비율. 리랭커는 이 후보 안에서 순서만 바꾸므로 리랭킹의 상한선
  - **MRR**: 정답이 몇 번째로 나왔는지까지 반영 (1등 1점, 2등 0.5점, 3등 0.33점 …)

**결과** (Google Colab T4 GPU, 고정 길이 청킹)

| 실험 | 임베딩 모델 | 리랭킹 | Hit@1 | Hit@3 | Hit@10 | MRR | 질문당 시간 |
|------|------------|--------|-------|-------|--------|-----|------------|
| A | all-MiniLM-L6-v2 | X | 1.7% | 3.7% | 7.3% | 0.031 | 1.5ms |
| B | all-MiniLM-L6-v2 | O | 7.3% | 7.7% | 8.3% | 0.076 | 177ms |
| C | BAAI/bge-m3 | X | 95.3% | 98.3% | 100% | 0.971 | 6.9ms |
| D | BAAI/bge-m3 | O | 100% | 100% | 100% | 1.000 | 660ms |

**해석**
- 기존 영어 임베딩 모델(A)은 한국어 질문에서 정답 문단을 거의 찾지 못했습니다. 다국어 모델 bge-m3로 바꾸자 Hit@3가 3.7%에서 98.3%로 올라, 이 결과를 근거로 챗봇의 임베딩 모델을 교체했습니다.
- 영어 모델에 리랭킹을 붙여도(B) Hit@3가 7.7%에 그쳤습니다. 후보 10개 안에 정답이 든 경우(Hit@10)가 8% 남짓이라, 순서만 바꾸는 리랭킹으로는 해결할 수 없는 문제였습니다.
- bge-m3에 리랭킹을 더하면(D) Hit@3가 1.7%p 오르지만 질문당 시간이 약 100배 늘어납니다. 서비스에서는 이 비용을 감수할 가치가 있는지 판단이 필요합니다.

**한계**
- KorQuAD 질문은 문단을 보고 만들어 단어가 많이 겹치고 문단도 300개뿐이라, 실제 사용자 질문보다 쉬운 평가입니다. C·D가 거의 만점이라 이후 개선(청킹 등)의 차이를 보려면 더 어려운 평가가 필요합니다.
- 실험마다 색인을 새로 만들고 ChromaDB가 근사 검색(HNSW)을 써서, 같은 임베딩 모델이어도 후보가 조금씩 달라질 수 있습니다. A와 B의 Hit@10 차이(7.3% vs 8.3%, 3문항)는 이 때문으로 보입니다. → 이후 스크립트를 고쳐, 리랭킹 전후를 **한 번 검색한 같은 후보**로 비교하도록 했습니다.

### 실험 2: 청킹 방식 비교 (문단 960개)

실험 1이 거의 만점이라 차이가 보이지 않아, KorQuAD 검증 데이터의 문단 전체(960개)로 평가를 어렵게 한 뒤 청킹 방식을 비교했습니다.

**배경:** 실험 1의 실패 사례를 분석하니, 영어 모델에서는 고정 길이로 자르고 남은 짧은 꼬리 청크(예: "아니아 지역 한정으로 발매됐다.")가 수많은 질문의 1등을 차지했습니다. 틀린 질문 289개의 1등 청크가 51종류뿐이었고, 한 청크는 혼자 30번 1등이었습니다. 단어 중간("오세아니아" → "아니아")에서 잘린 청크도 많았습니다.

**문장 단위 청킹:** 문장 경계에서만 자르고, 앞 청크의 마지막 문장 1개를 겹치며, 100자 미만의 꼬리는 앞 청크에 합칩니다 (`ingest.py`의 `chunk_by_sentence`).

| 청킹 | 리랭킹 | Hit@1 | Hit@3 | Hit@10 | MRR | 청크 수 | 60자 미만 청크 | 질문당 시간 |
|------|--------|-------|-------|--------|-----|--------|---------------|------------|
| 고정 길이 | X | 84.5% | 94.2% | 97.8% | 0.897 | 1,558 | 155 | 6.7ms |
| 고정 길이 | O | 94.6% | 97.7% | 97.8% | 0.961 | 1,558 | 155 | 608ms |
| 문장 단위 | X | 84.6% | 94.6% | 98.0% | 0.899 | 1,359 | 0 | 6.9ms |
| 문장 단위 | O | 94.8% | 97.8% | 98.0% | 0.963 | 1,359 | 0 | 617ms |

(임베딩 bge-m3, Google Colab T4 GPU)

**해석**
- 문장 단위 청킹의 검색 점수 차이(Hit@3 +0.4%p, 960문항 중 4문항)는 오차 범위 안이라, **검색 성능이 좋아졌다고 보지 않습니다.** bge-m3는 짧은 청크도 잘 이해해 꼬리 청크 문제가 거의 없었고, KorQuAD 문단이 짧아(평균 청크 1~2개) 자르는 방식의 영향이 작았던 것으로 보입니다.
- 그래도 짧은 꼬리 청크가 155개에서 0개로 사라지고 청크 수가 13% 줄었으며, 문장 중간에서 잘리지 않아 LLM에 넘기는 문맥이 깔끔해집니다. 검색 성능 손해 없이 얻는 이점이라 챗봇에 적용했습니다.
- 평가가 어려워지자 리랭킹 효과가 분명히 보였습니다. Hit@1이 84.5%에서 94.6%로 약 10%p 올랐고, 리랭킹 전후 Hit@10이 같아 같은 후보로 순서만 바뀌었음을 확인했습니다. 대신 질문당 시간이 약 90배 늘어납니다.

```bash
pip install pandas pyarrow
python eval_retrieval.py                                  # A
python eval_retrieval.py --rerank                         # B
python eval_retrieval.py --embed BAAI/bge-m3              # C
python eval_retrieval.py --embed BAAI/bge-m3 --rerank     # D

# 실험 2 (--n-docs를 크게 주면 검증 데이터의 문단 전체를 사용)
python eval_retrieval.py --n-docs 5000 --embed BAAI/bge-m3 --chunker fixed --rerank
python eval_retrieval.py --n-docs 5000 --embed BAAI/bge-m3 --chunker sentence --rerank
```

> KorQuAD 데이터는 라이선스(CC BY-ND)에 따라 저장소에 포함하지 않으며, 스크립트 실행 시 내려받습니다.

---

## 파일 구조

```
rag_chatbot/
├── ingest.py        # 문서 추가 / 삭제 / 조회
├── chatbot.py       # RAG 파이프라인 + 대화형 CLI
├── eval_retrieval.py # 검색 품질 평가 (KorQuAD 기반 Hit@k, MRR)
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
| `ingest.py` | `CHUNK_OVERLAP` | 청크 겹침 (fixed 방식) | 50자 |
| `ingest.py` | `CHUNK_METHOD` | 청킹 방식 (`fixed` / `sentence`) | sentence |
| `ingest.py` | `MIN_CHUNK_LEN` | 이보다 짧은 꼬리는 앞 청크에 합침 (sentence 방식) | 100자 |
| `ingest.py`, `chatbot.py` | `EMBED_MODEL` | 임베딩 모델 (두 파일이 같아야 함) | BAAI/bge-m3 |
| `chatbot.py` | `CANDIDATE_K` | 임베딩 검색 후보 수 | 10 |
| `chatbot.py` | `TOP_K` | LLM에 전달할 최종 청크 수 | 3 |
| `chatbot.py` | `RERANK_MODEL` | 리랭킹 모델 | bge-reranker-v2-m3 |
| `chatbot.py` | `GROQ_MODEL` | LLM 모델 | openai/gpt-oss-120b |

---

## 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 벡터 DB | ChromaDB |
| 임베딩 모델 | SentenceTransformers (BAAI/bge-m3) |
| 리랭킹 모델 | BAAI/bge-reranker-v2-m3 |
| LLM | Groq (openai/gpt-oss-120b) |
| 언어 | Python 3.10+ |

---

## 외부 LLM API 변화 대응

### Gemini API 한계 및 Groq 전환

개발 초기 Gemini API를 사용했으나 아래 문제로 Groq으로 전환했습니다.

| 문제 | 내용 |
|------|------|
| 한국 지역 제한 | 무료 티어 limit: 0 으로 설정되는 문제 발생 |
| 일일 요청 한도 | 429 RESOURCE_EXHAUSTED 오류 반복 |
| 분당 요청 한도 | 짧은 시간 내 여러 요청 시 차단 |

```
오류 예시:
429 RESOURCE_EXHAUSTED
* Quota exceeded for metric: generate_content_free_tier_requests, limit: 0
```

**Groq 선택 이유:** 한국에서 정상 작동, 넉넉한 무료 티어

### Groq 모델 지원 종료 대응

Groq가 `llama-3.3-70b-versatile`을 2026년 8월 16일에 지원 종료하면서 `404 model_not_found` 오류가 발생했습니다.
모델 이름을 `chatbot.py`의 `GROQ_MODEL` 한 곳에서만 관리하고 있어, Groq가 권장하는 대체 모델 `openai/gpt-oss-120b`로 한 줄만 바꿔 대응했습니다.
([Groq Deprecations](https://console.groq.com/docs/deprecations))

---

## 개선 가능한 점

> ✅ 완료: 한국어 임베딩 모델 교체 (all-MiniLM-L6-v2 → bge-m3), 문장 단위 청킹 적용 (결과는 [검색 품질 평가](#검색-품질-평가) 참고)

### 1. 멀티턴 대화 히스토리 유지

현재는 매 질문이 독립적으로 처리되어 이전 대화 내용을 기억하지 못합니다.

```python
# chatbot.py 개선안
chat_history = []  # 대화 히스토리 저장

def ask_groq(prompt: str) -> str:
    chat_history.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=chat_history  # 전체 히스토리 전달
    )

    answer = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": answer})
    return answer
```

### 2. 리랭킹 비용 줄이기
리랭킹은 Hit@1을 약 10%p 올리지만 질문당 시간이 약 90배 늘어납니다. 후보 수를 줄이거나 더 가벼운 리랭커를 써서 정확도와 속도의 균형점을 찾을 계획입니다.

### 3. 평가 고도화
- 검색뿐 아니라 LLM 답변 품질까지 평가
- KorQuAD처럼 문단과 단어가 겹치는 질문이 아니라, 실제 사용자 질문에 가까운 평가셋 만들기

### 4. 기타
- PDF, DOCX 등 다양한 파일 형식 지원
- 웹 UI 추가 (Streamlit 등)

---

## 라이선스

MIT License
