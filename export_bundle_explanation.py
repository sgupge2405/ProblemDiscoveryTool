# export_bundle_explanation.py
# 根拠付与モジュール専用：最終IRから ir.json と report.md を生成し，ZIPでエクスポートする
# - ir.jsonl（サーバ保存）とは別の「ローカル保存用」成果物
# - 対話ログは含めない（要求仕様）

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _now_jst_iso() -> str:
    # JST固定（+09:00）で十分
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")


def _safe(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, "_")
    return s


def build_ir_json(new_ir: Dict[str, Any]) -> Dict[str, Any]:
    """根拠付与モジュールのプロンプト（objective_card + *_explanation配列）に合わせた ir.json を生成する"""
    oc = (new_ir or {}).get("objective_card") or {}

    def s(key: str) -> str:
        return str(oc.get(key, "") or "")

    def a(key: str) -> List[str]:
        arr = oc.get(key) or []
        return [str(x) for x in arr if str(x).strip()]

    return {
        "objective_card": {
            "target_user": s("target_user"),
            "target_user_explanation": a("target_user_explanation"),

            "problem": s("problem"),
            "problem_explanation": a("problem_explanation"),

            "contradiction": s("contradiction"),
            "contradiction_explanation": a("contradiction_explanation"),

            "objective": s("objective"),
            "objective_explanation": a("objective_explanation"),

            "value": s("value"),
            "value_explanation": a("value_explanation"),

            "background": s("background"),
            "extras": a("extras"),
        }
    }


def build_report_md(ir_json: Dict[str, Any]) -> str:
    """人間可読性重視：各項目を『概要』『説明』で縦に並べる（対話ログなし）"""
    oc = (ir_json or {}).get("objective_card") or {}

    def section(title: str, value_key: str, expl_key: str) -> str:
        v = oc.get(value_key, "") or ""
        expl = oc.get(expl_key) or []
        expl_lines = "\n".join([f"- {x}" for x in expl]) if expl else "（なし）"
        return (
            f"## {title}\n"
            f"**概要**\n{v if v else '（なし）'}\n\n"
            f"**説明**\n{expl_lines}\n"
        )

    md: List[str] = []
    md.append("# 根拠付与モジュール 出力\n\n")

    md.append(section("対象ユーザー（target_user）", "target_user", "target_user_explanation"))
    md.append("\n---\n\n")
    md.append(section("課題（problem）", "problem", "problem_explanation"))
    md.append("\n---\n\n")
    md.append(section("矛盾（contradiction）", "contradiction", "contradiction_explanation"))
    md.append("\n---\n\n")
    md.append(section("目的（objective）", "objective", "objective_explanation"))
    md.append("\n---\n\n")
    md.append(section("価値（value）", "value", "value_explanation"))
    md.append("\n---\n\n")

    # 補足（background + extras）
    bg = oc.get("background", "") or ""
    extras = oc.get("extras") or []
    extras_lines = "\n".join([f"- {x}" for x in extras]) if extras else "（なし）"

    md.append("## 補足\n")
    md.append("**概要**\n")
    md.append(f"{bg if bg else '（なし）'}\n\n")
    md.append("**説明**\n")
    md.append(f"{extras_lines}\n")

    return "".join(md)


def build_zip_explanation(
    *,
    new_ir: Dict[str, Any],
    session_id: str,
    revision: int,
    stage: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, str]:
    """ZIP（ir.json + report.md + meta.json）を生成して bytes とファイル名を返す"""
    run_id = f"{_safe(session_id)}_rev{_safe(str(revision))}_{_safe(stage)}"
    base = f"runs/{run_id}/"

    ir_json = build_ir_json(new_ir)
    report_md = build_report_md(ir_json)

    meta: Dict[str, Any] = {
        "tool": "app_evidence_grounding",
        "run_mode": "ui",
        "created_at": _now_jst_iso(),
        "session_id": session_id,
        "revision": revision,
        "stage": stage,
    }
    if extra_meta:
        meta["extra"] = extra_meta

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(base + "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        zf.writestr(base + "ir.json", json.dumps(ir_json, ensure_ascii=False, indent=2))
        zf.writestr(base + "report.md", report_md)

    buf.seek(0)
    return buf.getvalue(), f"{run_id}_bundle.zip"
