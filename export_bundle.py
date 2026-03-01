# export_bundle.py
# 評価実験用データ（chat_log.jsonl / chat_log.md / ir_end.json / meta.json）を
# 一意なrun_idフォルダにまとめてZIP化するユーティリティ

from __future__ import annotations

import io
import json
import secrets
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Tuple


def make_run_id(prefix: str = "run") -> str:
    """
    run_id を生成する。
    例: run_20260104_113210_a8f3c1
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rnd = secrets.token_hex(3)  # 6 hex
    return f"{prefix}_{ts}_{rnd}"


def build_chat_log_jsonl(history: List[Tuple[str, str]], session_id: str) -> str:
    """
    対話ログを JSONL として生成する。
    1行=1発話（role=interviewer/interviewee）
    """
    lines: List[str] = []
    for i, (q, a) in enumerate(history, start=1):
        lines.append(
            json.dumps(
                {
                    "session_id": session_id,
                    "turn": i,
                    "role": "interviewer",
                    "text": q,
                    "created_utc": datetime.utcnow().isoformat() + "Z",
                },
                ensure_ascii=False,
            )
        )
        lines.append(
            json.dumps(
                {
                    "session_id": session_id,
                    "turn": i,
                    "role": "interviewee",
                    "text": a,
                    "created_utc": datetime.utcnow().isoformat() + "Z",
                },
                ensure_ascii=False,
            )
        )

    return "\n".join(lines) + ("\n" if lines else "")


def build_chat_log_md(history: List[Tuple[str, str]], session_id: str, stop_reason: str) -> str:
    """
    対話ログを Markdown として生成する（人間確認用）。
    """
    md_lines: List[str] = [
        "# Auto Interview Log",
        f"- session_id: {session_id}",
        f"- created_utc: {datetime.utcnow().isoformat()}Z",
        "",
    ]
    for i, (q, a) in enumerate(history, start=1):
        md_lines += [
            f"## Turn {i}",
            f"**Q:** {q}",
            "",
            f"**A:** {a}",
            "",
        ]
    md_lines += ["---", f"stop_reason: {stop_reason}", f"total_turns: {len(history)}"]
    return "\n".join(md_lines)


def build_meta(
    *,
    run_id: str,
    session_id: str,
    stop_reason: str,
    tool: str = "app_interview_auto",
    admin_rag: bool = False,
    run_mode: str = "step",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    実行条件のメタ情報を生成する。
    評価appで「同一条件か」を確認するための最小限を想定。
    """
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "session_id": session_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "tool": tool,
        "stop_reason": stop_reason,
        "admin_rag": bool(admin_rag),
        "run_mode": run_mode,
        "note": "",
    }
    if extra:
        meta["extra"] = extra
    return meta


def build_zip_bundle(
    *,
    run_id: str,
    chat_jsonl: str,
    chat_md: str,
    ir_end: Dict[str, Any],
    meta: Dict[str, Any],
) -> bytes:
    """
    ZIP（bytes）を生成する。
    ZIP内部は runs/<run_id>/ 以下に固定。
    """
    buf = io.BytesIO()
    base = f"runs/{run_id}/"
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(base + "chat_log.jsonl", chat_jsonl)
        z.writestr(base + "chat_log.md", chat_md)
        z.writestr(base + "ir_end.json", json.dumps(ir_end, ensure_ascii=False, indent=2))
        z.writestr(base + "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    return buf.getvalue()


def build_run_zip(
    *,
    history: List[Tuple[str, str]],
    session_id: str,
    stop_reason: str,
    ir_end: Dict[str, Any],
    tool: str = "app_interview_auto_1216",
    admin_rag: bool = False,
    run_mode: str = "step",
    extra_meta: Dict[str, Any] | None = None,
) -> Tuple[bytes, str, str]:
    """
    1回分の実行データをまとめてZIP化するワンストップ関数。

    Returns:
        zip_bytes, run_id, zip_name
    """
    run_id = make_run_id()
    chat_jsonl = build_chat_log_jsonl(history, session_id)
    chat_md = build_chat_log_md(history, session_id, stop_reason)
    meta = build_meta(
        run_id=run_id,
        session_id=session_id,
        stop_reason=stop_reason,
        tool=tool,
        admin_rag=admin_rag,
        run_mode=run_mode,
        extra=extra_meta,
    )
    zip_bytes = build_zip_bundle(
        run_id=run_id,
        chat_jsonl=chat_jsonl,
        chat_md=chat_md,
        ir_end=ir_end,
        meta=meta,
    )
    return zip_bytes, run_id, f"{run_id}.zip"
