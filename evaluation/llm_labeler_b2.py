# evaluation/llm_labeler_b2.py
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

from chains import call_llm, MODEL_NAME
from evaluation.text_normalize import normalize_text


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        cand = s[start : end + 1].strip()
        if cand.startswith("{") and cand.endswith("}"):
            return cand
    return None


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _load_deepdive_keys(categories_path: Path) -> List[str]:
    cfg = json.loads(categories_path.read_text(encoding="utf-8"))
    return [c["key"] for c in cfg["deepdive_categories"]]


@dataclass
class LabelerB2Config:
    prompt_path: Path
    categories_path: Path
    temperature: float = 0.2
    max_retries: int = 2


class LLMLabelerB2:
    """
    tool_b2_deepdive.txt を用いて、質問文→深掘りカテゴリ(単一) を付与する。
    """

    def __init__(self, cfg: LabelerB2Config):
        self.cfg = cfg
        self.prompt_template = cfg.prompt_path.read_text(encoding="utf-8")
        self.categories_json_text = cfg.categories_path.read_text(encoding="utf-8")

        self.deepdive_keys = _load_deepdive_keys(cfg.categories_path)

        self.prompt_sha256 = _sha256_text(self.prompt_template)
        self.categories_sha256 = _sha256_text(self.categories_json_text)

    def build_prompt(self, text: str) -> str:
        return (
            self.prompt_template
            .replace("{CATEGORIES_JSON}", self.categories_json_text)
            .replace("{TEXT}", text)
        )

    def _validate_category(self, cat: Any) -> Optional[str]:
        if not isinstance(cat, str):
            return None
        cat = cat.strip()
        return cat if cat in self.deepdive_keys else None

    def label_deepdive_category(
        self,
        *,
        sample_id: str,
        turn: int,
        question_text: Any,
        meta: Dict[str, Any],
        out_jsonl: Path,
        err_jsonl: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        1質問を深掘りカテゴリ(単一)に分類。
        成功：{"category": str, "rationale": str}
        失敗：None（errors.jsonlに記録）
        """
        text = normalize_text(question_text)
        if not text:
            return {"category": "Extras", "rationale": ""}

        prompt = self.build_prompt(text)

        last_raw = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = call_llm(
                    MODEL_NAME,
                    system_prompt=prompt,
                    user_text="",
                    temperature=self.cfg.temperature,
                )
                last_raw = resp

                js = _extract_json_object(resp)
                if js is None:
                    raise ValueError("JSON_NOT_FOUND")

                obj = _safe_json_loads(js)
                if obj is None:
                    raise ValueError("JSON_PARSE_FAILED")

                cat = self._validate_category(obj.get("category"))
                if cat is None:
                    raise ValueError("INVALID_CATEGORY")

                rationale = obj.get("rationale") or ""
                if not isinstance(rationale, str):
                    rationale = ""

                record = {
                    "sample_id": sample_id,
                    "kind": "deepdive_category",
                    "turn": int(turn),
                    "input": text,
                    "output": {"category": cat, "rationale": rationale},
                    "meta": meta,
                    "provenance": {
                        "model": MODEL_NAME,
                        "prompt_name": self.cfg.prompt_path.name,
                        "prompt_sha256": self.prompt_sha256,
                        "categories_name": self.cfg.categories_path.name,
                        "categories_sha256": self.categories_sha256,
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                out_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                return {"category": cat, "rationale": rationale}

            except Exception as e:
                if attempt >= self.cfg.max_retries:
                    err = {
                        "sample_id": sample_id,
                        "kind": "deepdive_category",
                        "turn": int(turn),
                        "error": str(e),
                        "raw_response": last_raw,
                        "meta": meta,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    err_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    with err_jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                    return None
                continue
