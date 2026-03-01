# storage.py — IRをJSONLで保存／sessionごとの履歴取得

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent / "data"
IR_PATH = DATA_DIR / "ir.jsonl"
DATA_DIR.mkdir(parents=True, exist_ok=True)
IR_PATH.touch(exist_ok=True)

def save_ir_jsonl(ir: Dict[str, Any], path: Path = IR_PATH) -> Path:
    """
    IR(dict) を JSON Lines 形式で1行追記して保存する。
    revision / stage / log.saved_at が無ければここで埋める。
    """
    ir = dict(ir)  # shallow copy

    # ここで今回決めたフィールドを補完する
    ir.setdefault("revision", 1)
    ir.setdefault("stage", "extraction")
    ir.setdefault("log", {})
    ir["log"].setdefault("saved_at", datetime.utcnow().isoformat() + "Z")

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ir, ensure_ascii=False) + "\n")
    return path


def load_ir_history_for_session(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    指定された session_id の IR 履歴を data/ir.jsonl から読み出す。
    - 保存日時の降順で最大 limit 件を返す。
    """
    items: List[Dict[str, Any]] = []
    if not IR_PATH.exists():
        return items

    with IR_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            sid = obj.get("session_id") or obj.get("objective_card", {}).get("session_id")
            if sid == session_id:
                items.append(
                    {
                        "mode": "IR",
                        "ir": obj,
                        "ts": (obj.get("log", {}) or {}).get("saved_at", ""),
                    }
                )

    items.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return items[:limit]