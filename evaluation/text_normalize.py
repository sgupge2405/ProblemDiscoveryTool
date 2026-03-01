# evaluation/text_normalize.py
from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(x: Any) -> str:
    """
    LLM投入前の正規化（分類品質優先）
    - None -> ""
    - list -> 改行結合
    - unicode正規化（NFKC）
    - 改行の統一、制御文字除去、連続空白圧縮
    """
    if x is None:
        s = ""
    elif isinstance(x, list):
        s = "\n".join(str(v).strip() for v in x if v is not None and str(v).strip())
    else:
        s = str(x).strip()

    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()
