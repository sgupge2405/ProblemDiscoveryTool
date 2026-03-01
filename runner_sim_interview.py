from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from chains import run_evidence_attach, run_evidence_finalize, run_persona_answer
from storage import load_ir_history_for_session, save_ir_jsonl


# -----------------------------
# ユーティリティ
# -----------------------------
STOP_TOKENS = {"@@", "＠＠"}

def now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def is_stop(s: str) -> bool:
    return normalize_ws(s) in STOP_TOKENS

def extract_question(out: Dict[str, Any]) -> str:
    for k in ("question", "next_question", "q", "text", "content"):
        v = out.get(k)
        if isinstance(v, str) and v.strip():
            candidate = v.strip()
            break
    else:
        candidate = ""

    if not candidate:
        return ""

    lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
    if not lines:
        return ""

    for ln in lines:
        m = re.match(r"^(質問|Q)\s*[:：]\s*(.+)$", ln)
        if m:
            return m.group(2).strip()

    for ln in lines:
        if ln.endswith("？") or ln.endswith("?"):
            return ln

    return lines[0]


def build_context_for_interviewer(oc: Dict[str, Any], history: List[Tuple[str, str]]) -> str:
    extras = oc.get("extras") or []
    hist_text = ""
    if history:
        recent = history[-10:]
        chunks = []
        for i, (q, a) in enumerate(recent, start=max(1, len(history) - len(recent) + 1)):
            chunks.append(f"[{i}] Q: {q}\n    A: {a}")
        hist_text = "\n".join(chunks)

    return (
        "【状況メモ】\n"
        f"対象ユーザー: {oc.get('target_user','')}\n"
        f"課題: {oc.get('problem','')}\n"
        f"矛盾: {oc.get('contradiction','')}\n"
        f"目的: {oc.get('objective','')}\n"
        f"価値: {oc.get('value','')}\n"
        f"背景: {oc.get('background','')}\n"
        f"補足: {', '.join(extras) if extras else ''}\n"
        "\n"
        "【これまでの会話（直近）】\n"
        f"{hist_text if hist_text else '（まだ会話なし）'}\n"
        "\n"
        "【指示】\n"
        "次に尋ねるべき質問を1つだけ生成してください。"
        " もう十分に深掘りでき、これ以上質問が不要だと判断した場合は @@ のみを出力してください。\n"
    )


def build_context_for_persona(oc: Dict[str, Any], question: str) -> str:
    extras = oc.get("extras") or []
    return (
        "【状況メモ】\n"
        f"対象ユーザー: {oc.get('target_user','')}\n"
        f"課題: {oc.get('problem','')}\n"
        f"矛盾: {oc.get('contradiction','')}\n"
        f"目的: {oc.get('objective','')}\n"
        f"価値: {oc.get('value','')}\n"
        f"背景: {oc.get('background','')}\n"
        f"補足: {', '.join(extras) if extras else ''}\n"
        "\n"
        "【質問】\n"
        f"{question}\n"
    )


def load_base_ir_and_objective_card(session_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    session_id に紐づく「第1モジュールIR（extraction）」を優先して取得し，
    (base_ir, objective_card) を返す。
    """
    items = load_ir_history_for_session(session_id, limit=200)

    candidate = None
    for row in items:
        ir = row.get("ir")
        if isinstance(ir, dict) and ir.get("stage") == "extraction":
            candidate = ir
            break
    if candidate is None:
        for row in items:
            ir = row.get("ir")
            if isinstance(ir, dict):
                candidate = ir
                break

    if candidate is None:
        raise RuntimeError("IR が見つかりませんでした。session_id を確認してください。")

    oc = candidate.get("objective_card")
    if not isinstance(oc, dict):
        raise RuntimeError("objective_card が見つかりませんでした。IR形式を確認してください。")

    return candidate, oc


def next_revision_for_session(session_id: str) -> int:
    items = load_ir_history_for_session(session_id, limit=500)
    revs = []
    for row in items:
        ir = row.get("ir")
        if isinstance(ir, dict) and isinstance(ir.get("revision"), int):
            revs.append(ir["revision"])
    return (max(revs) + 1) if revs else 1


def build_history_text_for_finalize(history: List[Tuple[str, str]]) -> str:
    # finalize には “全ログ” を渡す（root_explanation の根拠用）
    lines = []
    for i, (q, a) in enumerate(history, start=1):
        lines.append(f"[{i}] インタビュアー：{q}")
        lines.append(f"    ユーザー：{a}")
    return "\n".join(lines) if lines else "（まだ会話なし）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_id", required=True, help="第1モジュールで保存された session_id")
    ap.add_argument("--max_turns", type=int, default=20, help="安全装置の最大ターン数（推奨: 20）")
    ap.add_argument("--print_live", type=int, default=1, help="ターミナルに進行を表示する(1) / しない(0)")
    ap.add_argument("--save_ir", type=int, default=1, help="最後にIR確定保存する(1) / しない(0)")
    args = ap.parse_args()

    session_id: str = args.session_id
    max_turns: int = args.max_turns
    print_live: bool = bool(args.print_live)
    save_ir: bool = bool(args.save_ir)

    base_ir, oc = load_base_ir_and_objective_card(session_id)

    logs_dir = Path("data") / "logs"
    ensure_dir(logs_dir)
    tag = now_tag()
    jsonl_path = logs_dir / f"{session_id}_{tag}.jsonl"
    md_path = logs_dir / f"{session_id}_{tag}.md"

    history: List[Tuple[str, str]] = []  # (Q, A)

    md_lines: List[str] = []
    md_lines.append(f"# Auto Interview Log")
    md_lines.append(f"- session_id: {session_id}")
    md_lines.append(f"- max_turns: {max_turns}")
    md_lines.append(f"- created_utc: {datetime.utcnow().isoformat()}Z")
    md_lines.append("")

    stop_reason = "max_turns"

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for turn in range(1, max_turns + 1):
            interviewer_ctx = build_context_for_interviewer(oc, history)
            q_out = run_evidence_attach(interviewer_ctx)
            question = extract_question(q_out)

            if not question:
                question = "（質問の生成に失敗しました。どの点を深掘りすべきですか？）"

            if print_live:
                print(f"\n[Turn {turn}] Q: {question}")

            if is_stop(question):
                stop_reason = "interviewer_stop(@@)"
                if print_live:
                    print(f"[End] stop_reason={stop_reason}")
                md_lines.append(f"## Turn {turn}")
                md_lines.append(f"**Q:** @@")
                md_lines.append(f"**A:** （終了）")
                md_lines.append("")
                jf.write(json.dumps({
                    "session_id": session_id,
                    "turn": turn,
                    "question": "@@",
                    "answer": "",
                    "stop_reason": stop_reason,
                    "created_utc": datetime.utcnow().isoformat() + "Z",
                }, ensure_ascii=False) + "\n")
                break

            persona_ctx = build_context_for_persona(oc, question)
            a_out = run_persona_answer(persona_ctx)
            answer = (a_out.get("text") or "").strip()

            if print_live:
                print(f"[Turn {turn}] A: {answer[:200]}{'...' if len(answer) > 200 else ''}")

            history.append((question, answer))

            jf.write(json.dumps({
                "session_id": session_id,
                "turn": turn,
                "question": question,
                "answer": answer,
                "created_utc": datetime.utcnow().isoformat() + "Z",
            }, ensure_ascii=False) + "\n")

            md_lines.append(f"## Turn {turn}")
            md_lines.append(f"**Q:** {question}")
            md_lines.append("")
            md_lines.append(f"**A:** {answer}")
            md_lines.append("")

    # md 保存（最後に一括）
    md_lines.append(f"---")
    md_lines.append(f"stop_reason: {stop_reason}")
    md_lines.append(f"total_turns: {len(history)}")
    md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write("\n".join(md_lines))

    print("\n[Saved Logs]")
    print(f"- {jsonl_path}")
    print(f"- {md_path}")

    # -----------------------------
    # ★ここが追加ポイント：IR確定→新規保存
    # -----------------------------
    if save_ir:
        history_text = build_history_text_for_finalize(history)

        finalize_input = (
            "【第1モジュールIR（objective_card を含む）】\n"
            f"{base_ir}\n\n"
            "【根拠付与インタビューの全ログ】\n"
            f"{history_text}\n\n"
            "【指示】\n"
            "上のログを根拠として，objective_card の各項目に対応する root_explanation を追記し，"
            "更新後の IR(JSON) を1つだけ出力してください．\n"
            "- 出力は JSON のみ（前後説明禁止）\n"
            "- target_user/problem/contradiction/objective/value の各 *_root_explanation は配列で，"
            "  今回追加する要約（根拠）を必要な分だけ append してください．\n"
            "- background は必要なら更新してよい\n"
            "- extras は必要なら追加してよい\n"
        )

        fin = run_evidence_finalize(finalize_input)

        if fin.get("mode") != "IR" or not isinstance(fin.get("ir"), dict):
            raise RuntimeError("IR(JSON)の確定生成に失敗しました（mode!=IR）。")

        new_ir = fin["ir"]

        # session_id は必ず継承（固定IDは runner の引数で制御）
        new_ir["session_id"] = session_id
        new_ir["stage"] = "evidence"
        new_ir["revision"] = next_revision_for_session(session_id)

        save_ir_jsonl(new_ir)

        print("\n[Saved IR]")
        print(f"- session_id: {new_ir['session_id']}")
        print(f"- stage: {new_ir['stage']}")
        print(f"- revision: {new_ir['revision']}")



if __name__ == "__main__":
    main()
