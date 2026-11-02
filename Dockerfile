# Hugging Face Spaces(Docker)용 이미지
# 빌드할 때 문서를 미리 DB에 넣고 모델도 내려받아 두어, 서버가 켜질 때 기다리는 시간을 줄인다.
FROM python:3.11-slim

# Spaces는 사용자 ID 1000으로 컨테이너를 실행한다
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/app/.cache \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# GPU가 없는 서버라 용량이 작은 CPU 전용 PyTorch를 먼저 설치
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# docs 폴더의 취업규칙을 DB에 넣고(임베딩 모델도 이때 내려받음), 리랭커 모델도 미리 받아 둔다
RUN python ingest.py && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
