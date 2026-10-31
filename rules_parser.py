"""
rules_parser.py — 취업규칙처럼 "제○조(제목)"으로 이뤄진 규정 문서를 조항 단위로 나누는 모듈

고용노동부 표준취업규칙(2026년 배포) 파일 구성
- 안내문 → 목차 → [일반 근로자용 본문 제1조~제98조] → 부칙 → 별첨 → 단시간 근로자용 …
- 조항마다 본문 뒤에 담당자용 작성 해설("[필수] …", "☞ (참고) …")이 붙어 있다.

이 모듈은 일반 근로자용 본문만 골라, 조항마다 해설을 뺀 본문과 장·조 정보를 돌려준다.
"""

import re

ARTICLE_RE = re.compile(r"^제(\d+)조(?:의\d+)?\(([^)]*)\)")   # 제33조(연차유급휴가)
CHAPTER_RE = re.compile(r"^제\d+장\s")                          # 제6장 휴일․휴가
NOTE_MARKERS = ("[필수", "[선택", "☞", "◈", "※")               # 여기서부터는 작성 해설


def _body_range(text: str) -> str:
    """목차를 건너뛰고, 일반 근로자용 본문(제1조 ~ 부칙 전)만 잘라 낸다."""
    starts = [m.start() for m in re.finditer(r"(?m)^제1조\(목적\) (?!\[)", text)]
    if not starts:
        raise ValueError("본문의 제1조(목적)를 찾지 못했습니다")
    start = starts[0]                                  # 목차의 "제1조(목적) [필수]"는 제외됨
    end_match = re.search(r"(?m)^부\s*칙\s*$", text[start:])
    end = start + end_match.start() if end_match else len(text)
    return text[start:end]


def parse_articles(text: str, doc_name: str = "취업규칙") -> list[dict]:
    """조항 목록을 돌려준다. 각 조항: {"label", "article", "title", "chapter", "text"}"""
    body = _body_range(text)
    articles, chapter, current, in_note = [], "", None, False

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if CHAPTER_RE.match(line):
            chapter = re.sub(r"\s+", " ", line)
            in_note = True        # 장 제목 뒤의 안내문(◈ …)도 조항 본문이 아니다
            continue
        m = ARTICLE_RE.match(line)
        if m:
            current = {
                "article": f"제{m.group(1)}조",
                "title": m.group(2).strip(),
                "chapter": chapter,
                "lines": [line],
            }
            articles.append(current)
            in_note = False
            continue
        if line.startswith(NOTE_MARKERS):
            in_note = True
        if current is not None and not in_note:
            current["lines"].append(line)

    results = []
    for a in articles:
        label = f"{doc_name} {a['article']}({a['title']})"
        results.append({
            "label": label,
            "article": a["article"],
            "title": a["title"],
            "chapter": a["chapter"],
            "text": "\n".join(a.pop("lines")),
        })
    return results


if __name__ == "__main__":
    import sys
    from hwp_loader import read_hwp_text

    arts = parse_articles(read_hwp_text(sys.argv[1]), "표준취업규칙")
    print(f"조항 {len(arts)}개")
    for a in arts[:3]:
        print(a["label"], "|", a["chapter"])
        print(a["text"][:200], "\n")
