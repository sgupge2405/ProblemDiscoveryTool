# evaluation/llm_labeler.py
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

from chains import call_llm, MODEL_NAME  # 既存のOpenAI設定を再利用
from evaluation.text_normalize import normalize_text


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> Optional[str]:
    """
    返答がJSON以外を混ぜても、最初の { 〜 最後の } を拾う。
    """
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


def _load_meaning_keys(categories_path: Path) -> List[str]:
    cfg = json.loads(categories_path.read_text(encoding="utf-8"))
    return [c["key"] for c in cfg["meaning_categories"]]


@dataclass
class LabelerConfig:
    prompt_path: Path
    categories_path: Path
    temperature: float = 0.2
    max_retries: int = 2


class LLMLabeler:
    """
    tool_b_meaning.txt（スコア版）を想定したラベラー。
    出力JSONの scores をパースして返す。
    """

    def __init__(self, cfg: LabelerConfig):
        self.cfg = cfg
        self.prompt_template = cfg.prompt_path.read_text(encoding="utf-8")
        self.categories_json_text = cfg.categories_path.read_text(encoding="utf-8")

        self.meaning_keys = _load_meaning_keys(cfg.categories_path)

        # provenance
        self.prompt_sha256 = _sha256_text(self.prompt_template)
        self.categories_sha256 = _sha256_text(self.categories_json_text)

    def build_prompt(self, text: str) -> str:
        return (
            self.prompt_template
            .replace("{CATEGORIES_JSON}", self.categories_json_text)
            .replace("{TEXT}", text)
        )

    def _validate_scores(self, scores: Any) -> Optional[Dict[str, int]]:
        """
        scores:
          - dict[str, int] であること
          - 12カテゴリすべてを含むこと
          - 各値が 0〜3 の int であること
        """
        if not isinstance(scores, dict):
            return None

        out: Dict[str, int] = {}
        for k in self.meaning_keys:
            if k not in scores:
                return None
            v = scores[k]
            if isinstance(v, bool):
                return None
            if not isinstance(v, int):
                # "2" のような文字列が来た場合に備え、int変換を試す
                try:
                    v = int(v)
                except Exception:
                    return None
            if v < 0 or v > 3:
                return None
            out[k] = v

        # 余計なキーが入ってもよいが、ここでは無視する
        return out

    def label_meaning_scores(
        self,
        *,
        sample_id: str,
        text_id: str,
        raw_text: Any,
        meta: Dict[str, Any],
        out_jsonl: Path,
        err_jsonl: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        意味カテゴリ（スコア）付与（1テキスト）。
        成功したら {"scores": {...}, "rationales": {...}} を返す。
        失敗したら None（errors.jsonlに記録）。
        """
        text = normalize_text(raw_text)
        if not text:
            # 空はスキップ（必要なら errors に落としてもよい）
            return {"scores": {k: 0 for k in self.meaning_keys}, "rationales": {}}

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

                scores = self._validate_scores(obj.get("scores"))
                if scores is None:
                    raise ValueError("INVALID_SCORES")

                rationales = obj.get("rationales") or {}
                if not isinstance(rationales, dict):
                    rationales = {}

                # write jsonl
                record = {
                    "sample_id": sample_id,
                    "kind": "meaning_scores",
                    "text_id": text_id,
                    "input": text,
                    "output": {
                        "scores": scores,
                        "rationales": rationales,
                    },
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

                return {"scores": scores, "rationales": rationales}

            except Exception as e:
                if attempt >= self.cfg.max_retries:
                    err = {
                        "sample_id": sample_id,
                        "kind": "meaning_scores",
                        "text_id": text_id,
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
