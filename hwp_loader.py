"""
hwp_loader.py — 한글(.hwp, HWP 5.0) 문서에서 본문 텍스트를 꺼내는 모듈

HWP 5.0 파일 구조 (한컴 공개 문서 "한글 문서 파일 형식 5.0" 기준)
- 파일 전체는 OLE 복합 파일이다 (안에 여러 개의 '스트림'이 들어 있는 상자 구조).
- 본문은 BodyText/Section0, Section1 ... 스트림에 나뉘어 저장된다.
- FileHeader 스트림의 속성 값 첫 비트가 1이면 각 스트림이 zlib(raw deflate)으로 압축돼 있다.
- 스트림 안은 '레코드'의 연속이다. 레코드 헤더 4바이트에 종류(tag)와 크기가 들어 있고,
  문단 글자는 HWPTAG_PARA_TEXT(67) 레코드에 UTF-16LE로 저장된다.
- 글자 사이사이에 표·그림 같은 '컨트롤'을 나타내는 특수 코드가 섞여 있어서 걸러내야 한다.

사용법:
    from hwp_loader import read_hwp_text
    text = read_hwp_text("docs/표준취업규칙(2026년, 배포).hwp")
"""

import struct
import zlib

import olefile

HWPTAG_PARA_TEXT = 67

# 컨트롤 문자: 아래 코드들은 뒤에 7글자(14바이트)짜리 부가 정보가 붙어 총 8글자를 차지한다.
EXTENDED_CTRL = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def _is_compressed(ole: olefile.OleFileIO) -> bool:
    header = ole.openstream("FileHeader").read()
    flags = struct.unpack_from("<I", header, 36)[0]  # 시그니처(32) + 버전(4) 다음이 속성
    return bool(flags & 1)


def _section_names(ole: olefile.OleFileIO) -> list[str]:
    names = ["/".join(p) for p in ole.listdir() if p[0] == "BodyText"]
    return sorted(names, key=lambda n: int(n.split("Section")[-1]))


def _paragraph_text(data: bytes) -> str:
    """PARA_TEXT 레코드의 내용을 글자로 바꾼다. 컨트롤 코드는 건너뛴다."""
    chars = []
    i = 0
    while i + 1 < len(data):
        code = struct.unpack_from("<H", data, i)[0]
        if code in EXTENDED_CTRL:
            if code == 9:        # 탭도 8글자를 차지하는 컨트롤이다
                chars.append("\t")
            i += 16              # 컨트롤 1글자 + 부가 정보 7글자
            continue
        i += 2
        if code in (10, 13):     # 줄바꿈, 문단 끝
            chars.append("\n")
        elif code in (30, 31):   # 묶음 빈칸, 고정폭 빈칸 → 일반 띄어쓰기로
            chars.append(" ")
        elif code < 32:          # 그 밖의 1글자짜리 컨트롤은 버린다
            continue
        else:
            chars.append(chr(code))
    return "".join(chars)


def _iter_records(stream: bytes):
    """스트림을 (tag, 내용) 레코드로 나눈다."""
    pos = 0
    while pos + 4 <= len(stream):
        header = struct.unpack_from("<I", stream, pos)[0]
        pos += 4
        tag = header & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:         # 크기가 4095 이상이면 다음 4바이트에 실제 크기가 있다
            size = struct.unpack_from("<I", stream, pos)[0]
            pos += 4
        yield tag, stream[pos:pos + size]
        pos += size


def read_hwp_text(path: str) -> str:
    """HWP 파일의 본문 텍스트 전체를 문단 단위 줄바꿈으로 이어 붙여 돌려준다."""
    with olefile.OleFileIO(path) as ole:
        compressed = _is_compressed(ole)
        paragraphs = []
        for name in _section_names(ole):
            stream = ole.openstream(name).read()
            if compressed:
                stream = zlib.decompress(stream, -15)  # raw deflate
            for tag, body in _iter_records(stream):
                if tag == HWPTAG_PARA_TEXT:
                    text = _paragraph_text(body).strip()
                    if text:
                        paragraphs.append(text)
    return "\n".join(paragraphs)


if __name__ == "__main__":
    import sys
    print(read_hwp_text(sys.argv[1])[:2000])
