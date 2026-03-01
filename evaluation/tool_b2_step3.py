# evaluation/tool_b2_step3.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

import pandas as pd

from evaluation.loader import TrialRecord
from evaluation.text_normalize import normalize_text

from openai import OpenAI


# ----------------------------
# Outputs
# ----------------------------
@dataclass
class ToolB2Step3Outputs:
    # sample_id × 指標（応答整合率など）
    alignment_df: pd.DataFrame

    # sample_id × turn × score/reason（詳細）
    alignment_detail_df: pd.DataFrame

    report: Dict[str, Any]


# ----------------------------
# Helpers
# ----------------------------
def _load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    LLM出力からJSONオブジェクトを抽出して返す。
    - 返答がJSON単体でなくても、最初の { ... } を拾う。
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty model output")

    # まず全文JSONとして試す
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 最初の {...} を拾う
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        cand = s[start : end + 1]
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from model output")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp_score_012(score: int) -> int:
    if score < 0:
        return 0
    if score > 2:
        return 2
    return score


# ----------------------------
# Main
# ----------------------------
def run_tool_b2_step3_response_alignment_012(
    trials: Sequence[TrialRecord],
    *,
    judge_prompt_path: Path,
    outputs_dir: Path,
    temperature: float = 0.2,
    max_retries: int = 2,
    model: Optional[str] = None,
) -> ToolB2Step3Outputs:
    """
    Tool B2 Step3（応答整合 0/1/2）:

    スコア定義（ユーザー合意）:
      0: 質問内容が回答に含まれていない／無関係
      1: 質問内容が回答に少しでも含まれている（背景・冗長・追加情報があってもOK）
      2: 質問内容に対する直接的・ほぼ一致の回答（完璧に近い）

    一致率（alignment_rate）:
      - 2は1に潰して「答えている率」を見たいので、score > 0 を一致とする

    出力（必須）:
      outputs_dir/
        - alignment.csv（sample_id単位の集計）
        - alignment_detail.csv（turn単位の詳細）
        - report.json
        - tool_b2_judge_labels.jsonl
        - tool_b2_judge_errors.jsonl
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    prompt_tmpl = _load_prompt(judge_prompt_path)

    # 必須出力ファイル
    alignment_csv = outputs_dir / "alignment.csv"
    alignment_detail_csv = outputs_dir / "alignment_detail.csv"
    report_json = outputs_dir / "report.json"

    # 監査ログ（jsonl）
    out_jsonl = outputs_dir / "tool_b2_judge_labels.jsonl"
    err_jsonl = outputs_dir / "tool_b2_judge_errors.jsonl"

    client = OpenAI()

    model_name = (
        model
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("OPENAI_CHAT_MODEL")
        or "gpt-4o-mini"
    )

    detail_rows: List[Dict[str, Any]] = []
    sum_rows: List[Dict[str, Any]] = []

    for t in trials:
        sample_id = t.zip_name
        meta = dict(t.meta or {})
        meta.update({"run_id": t.run_id})

        total_pairs = 0
        answered_pairs = 0  # score>0
        score0 = 0
        score1 = 0
        score2 = 0
        unanswered_pairs = 0
        judged_pairs = 0
        failed_pairs = 0

        for p in t.qa_pairs:
            q = normalize_text(p.question) if p.question else ""
            a = normalize_text(p.answer) if p.answer else ""
            if not q.strip():
                continue

            turn = int(p.turn)
            total_pairs += 1

            # 未回答
            if not a.strip():
                unanswered_pairs += 1
                score0 += 1
                detail_rows.append(
                    {
                        "sample_id": sample_id,
                        "turn": turn,
                        "question": q,
                        "answer": "",
                        "score": 0,
                        "is_aligned": 0,
                        "reason": "missing_answer",
                        "note": "missing_answer",
                    }
                )
                continue

            # プロンプト埋め込み（テンプレは {question} {answer} を想定）
            prompt = prompt_tmpl.format(question=q, answer=a)

            ok = False
            raw_text = ""
            score = 0
            reason = ""
            last_err: Optional[str] = None

            for attempt in range(max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        temperature=float(temperature),
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a strict evaluator. Output JSON only.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )
                    raw_text = (resp.choices[0].message.content or "").strip()
                    obj = _extract_json_object(raw_text)

                    score = _clamp_score_012(_safe_int(obj.get("score", 0), 0))
                    reason = str(obj.get("reason", "")).strip()

                    ok = True
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(0.2 * (attempt + 1))

            if not ok:
                failed_pairs += 1
                score = 0
                reason = f"judge_failed: {last_err}" if last_err else "judge_failed"
                _append_jsonl(
                    err_jsonl,
                    {
                        "sample_id": sample_id,
                        "turn": turn,
                        "question": q,
                        "answer": a,
                        "error": reason,
                        "meta": meta,
                    },
                )
                note = "judge_failed->score0"
            else:
                judged_pairs += 1
                _append_jsonl(
                    out_jsonl,
                    {
                        "sample_id": sample_id,
                        "turn": turn,
                        "score": score,
                        "reason": reason,
                        "question": q,
                        "answer": a,
                        "raw": raw_text,
                        "meta": meta,
                    },
                )
                note = ""

            if score == 0:
                score0 += 1
            elif score == 1:
                score1 += 1
            else:
                score2 += 1

            is_aligned = 1 if score > 0 else 0
            answered_pairs += is_aligned

            detail_rows.append(
                {
                    "sample_id": sample_id,
                    "turn": turn,
                    "question": q,
                    "answer": a,
                    "score": score,
                    "is_aligned": is_aligned,
                    "reason": reason,
                    "note": note,
                }
            )

        denom = total_pairs if total_pairs > 0 else 1

        alignment_rate = float(answered_pairs / denom)  # score>0率（2は1に潰して一致扱い）
        score_mean = float((0 * score0 + 1 * score1 + 2 * score2) / denom) if denom > 0 else 0.0
        score2_rate = float(score2 / denom) if denom > 0 else 0.0

        sum_rows.append(
            {
                "sample_id": sample_id,
                "total_pairs": int(total_pairs),
                "answered_pairs": int(answered_pairs),
                "alignment_rate": alignment_rate,
                "score0_pairs": int(score0),
                "score1_pairs": int(score1),
                "score2_pairs": int(score2),
                "score_mean": score_mean,
                "score2_rate": score2_rate,
                "unanswered_pairs": int(unanswered_pairs),
                "judged_pairs": int(judged_pairs),
                "failed_pairs": int(failed_pairs),
            }
        )

    alignment_df = pd.DataFrame(sum_rows)
    alignment_detail_df = pd.DataFrame(detail_rows)

    # ---- 必須CSV出力（Step5互換）----
    alignment_df.to_csv(alignment_csv, index=False, encoding="utf-8-sig")
    alignment_detail_df.to_csv(alignment_detail_csv, index=False, encoding="utf-8-sig")

    report: Dict[str, Any] = {
        "tool": "tool_b2_step3_response_alignment_012",
        "sample_id_policy": "zip_name",
        "target": "qa_pairs (question + answer)",
        "scoring": {
            "0": "question not covered / irrelevant (or missing answer)",
            "1": "question covered partially (extra background is OK)",
            "2": "question covered directly / almost equivalent",
            "alignment_rule": "aligned iff score > 0 (2 is counted as 1 for alignment_rate)",
        },
        "llm": {
            "model": model_name,
            "temperature": float(temperature),
            "max_retries": int(max_retries),
            "judge_prompt_path": str(judge_prompt_path),
        },
        "outputs": {
            "alignment_csv": str(alignment_csv),
            "alignment_detail_csv": str(alignment_detail_csv),
            "judge_labels_jsonl": str(out_jsonl),
            "judge_errors_jsonl": str(err_jsonl),
            "report_json": str(report_json),
        },
        "notes": [
            "Step3 was changed from category-alignment to response-alignment (0/1/2 + reason).",
            "alignment_rate uses score>0 (2 is merged into 1 for the ratio).",
            "alignment_detail.csv includes reasons for qualitative inspection.",
        ],
    }

    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return ToolB2Step3Outputs(
        alignment_df=alignment_df,
        alignment_detail_df=alignment_detail_df,
        report=report,
    )


# 互換用（旧名の関数を呼んでいる箇所が残っている場合に備えたエイリアス）
# UI側を更新したら、このエイリアスは消しても構いません。
def run_tool_b2_step3_alignment_multilabel(
    trials: Sequence[TrialRecord],
    *,
    judge_prompt_path: Path,
    outputs_dir: Path,
    temperature: float = 0.2,
    max_retries: int = 2,
    model: Optional[str] = None,
    **_ignored: Any,
) -> ToolB2Step3Outputs:
    """
    旧 Step3（カテゴリ一致）からの移行用エイリアス。
    UIが古い関数名を呼んでいても、新Step3（応答整合）へ誘導する。
    """
    return run_tool_b2_step3_response_alignment_012(
        trials,
        judge_prompt_path=judge_prompt_path,
        outputs_dir=outputs_dir,
        temperature=temperature,
        max_retries=max_retries,
        model=model,
    )
