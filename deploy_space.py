"""
deploy_space.py — Hugging Face Spaces에 배포할 파일만 골라 올린다.

사용법:
  pip install -U huggingface_hub
  python deploy_space.py 내아이디/rag-chatbot

- 처음 실행하면 Hugging Face 토큰(Write 권한)을 물어본다. 토큰은 화면에 보이지 않게 입력된다.
- 원본 HWP(docs/), 로컬 DB(chroma_db/), 평가 결과 CSV, .git 등은 올리지 않는다.
- Space용 README(space/README.md, 맨 위에 Space 설정이 있음)를 README.md로 올린다.
"""

import sys

from huggingface_hub import HfApi, login
from huggingface_hub.errors import LocalTokenNotFoundError

IGNORE = [
    ".git/*", "venv/*", "chroma_db/*", "docs/*", ".cache/*",
    "*__pycache__*", ".pytest_cache/*",
    "*.csv", "*.pyc", ".env",
    "bench_measure.py", "README.md", "space/*",
]


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("사용법: python deploy_space.py 내아이디/rag-chatbot")
    repo_id = sys.argv[1]

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except (LocalTokenNotFoundError, OSError):
        login()                       # 토큰 입력 (화면에 보이지 않음)
        user = api.whoami()["name"]
    print(f"로그인: {user}")

    api.upload_folder(folder_path=".", repo_id=repo_id, repo_type="space",
                      ignore_patterns=IGNORE, commit_message="배포: 앱 코드와 취업규칙 가공본")
    api.upload_file(path_or_fileobj="space/README.md", path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space", commit_message="배포: Space 설명")
    print(f"올리기 완료 → https://huggingface.co/spaces/{repo_id}")
    print("Space 화면의 'Building' 상태가 'Running'이 될 때까지 기다리세요 (10~20분).")


if __name__ == "__main__":
    main()
