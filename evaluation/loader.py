# evaluation/loader.py
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ChatRow:
    """jsonl 1行分（できるだけ元データを保持）"""
    session_id: Optional[str]
    turn: Optional[int]
    role: str
    text: str
    created_utc: Optional[str]
    raw: Dict[str, Any]


@dataclass
class QAPair:
    """turn単位のQ/A（揃わない場合もあるので片側Noneを許容）"""
    turn: int
    question: Optional[str]
    answer: Optional[str]


@dataclass
class TrialRecord:
    """1つのrun(zip)から復元した評価用レコード"""
    zip_name: str
    run_id: str
    meta: Dict[str, Any]
    ir: Dict[str, Any]
    chat_rows: List[ChatRow]
    qa_pairs: List[QAPair]

    # --- convenience ---
    @property
    def ir_text(self) -> str:
        # 評価Aで「IR全体」をベクトル化したい時のために提供
        return json.dumps(self.ir, ensure_ascii=False, sort_keys=True)

    @property
    def objective_card(self) -> Dict[str, Any]:
        # あなたのIR例では objective_card が本体なのでショートカット
        oc = self.ir.get("objective_card")
        return oc if isinstance(oc, dict) else {}

    @property
    def questions(self) -> List[str]:
        return [q for p in self.qa_pairs for q in ([p.question] if p.question else [])]

    @property
    def answers(self) -> List[str]:
        return [a for p in self.qa_pairs for a in ([p.answer] if p.answer else [])]


# -----------------------------
# Zip utilities
# -----------------------------

def _find_member(zf: zipfile.ZipFile, suffix: str) -> Optional[str]:
    """
    zip内から suffix で終わる最初のファイルパスを返す。
    export_bundle の runs/<run_id>/... を前提にしているが、階層は固定しない。
    """
    for name in zf.namelist():
        if name.endswith(suffix):
            return name
    return None


def _read_json(zf: zipfile.ZipFile, member: str) -> Dict[str, Any]:
    with zf.open(member) as f:
        return json.load(f)


def _read_text_lines(zf: zipfile.ZipFile, member: str) -> List[str]:
    with zf.open(member) as f:
        data = f.read().decode("utf-8", errors="replace")
    return data.splitlines()


# -----------------------------
# Parsing
# -----------------------------

def _parse_chat_jsonl_lines(lines: List[str]) -> List[ChatRow]:
    """
    chat_log.jsonl を ChatRow の配列へ。
    例の形式（session_id, turn, role, text, created_utc）を優先して読む。
    """
    rows: List[ChatRow] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                obj = {"_raw": obj}
        except json.JSONDecodeError:
            # 壊れている行があっても落とさない
            obj = {"_raw_line": line}

        session_id = obj.get("session_id") if isinstance(obj, dict) else None

        # turnは int に寄せる（なければNone）
        t = None
        if isinstance(obj, dict) and "turn" in obj:
            try:
                t = int(obj.get("turn"))
            except Exception:
                t = None

        role = ""
        if isinstance(obj, dict):
            role = str(obj.get("role") or obj.get("speaker") or obj.get("from") or "").strip()
        role_lower = role.lower()

        # 本文キーは text 優先（例データ準拠）
        text = ""
        if isinstance(obj, dict):
            text = (
                obj.get("text")
                or obj.get("content")
                or obj.get("message")
                or obj.get("utterance")
                or ""
            )
        text = str(text)

        created_utc = None
        if isinstance(obj, dict):
            cu = obj.get("created_utc") or obj.get("timestamp") or obj.get("created_at")
            created_utc = str(cu) if cu is not None else None

        rows.append(
            ChatRow(
                session_id=str(session_id) if session_id is not None else None,
                turn=t,
                role=role_lower,
                text=text,
                created_utc=created_utc,
                raw=obj if isinstance(obj, dict) else {"_raw": obj},
            )
        )
    return rows


def _build_qa_pairs(chat_rows: List[ChatRow]) -> List[QAPair]:
    """
    turn単位で interviewer/interviewee をペアリングする。
    - questioner/assistant/interviewer を「質問側」
    - interviewee/user/persona/answerer を「回答側」
    片側欠損があっても残す（後で評価の除外条件にできる）。
    """
    q_roles = {"interviewer", "questioner", "assistant"}
    a_roles = {"interviewee", "user", "persona", "answerer"}

    by_turn: Dict[int, Dict[str, Optional[str]]] = {}

    for r in chat_rows:
        if r.turn is None:
            # turnなしはペアリング困難なので無視（必要なら別途扱える）
            continue

        if r.turn not in by_turn:
            by_turn[r.turn] = {"q": None, "a": None}

        if r.role in q_roles and r.text:
            by_turn[r.turn]["q"] = r.text
        elif r.role in a_roles and r.text:
            by_turn[r.turn]["a"] = r.text
        else:
            # roleが未知でも、内容があり、まだ埋まってないなら保険として入れる
            #（ただし誤判定を避けたいならここをコメントアウト）
            if r.text:
                if by_turn[r.turn]["q"] is None:
                    by_turn[r.turn]["q"] = r.text
                elif by_turn[r.turn]["a"] is None:
                    by_turn[r.turn]["a"] = r.text

    pairs: List[QAPair] = []
    for t in sorted(by_turn.keys()):
        pairs.append(QAPair(turn=t, question=by_turn[t]["q"], answer=by_turn[t]["a"]))
    return pairs


def _infer_run_id(meta_member_path: str, meta: Dict[str, Any], zip_name: str) -> str:
    """
    run_id推定：
    - runs/<run_id>/meta.json 形式ならそこから抽出
    - meta に run_id/session_id があればそれを優先
    - なければ zip名
    """
    # metaに明示があるなら最優先
    for k in ("run_id", "session_id"):
        v = meta.get(k)
        if v:
            return str(v)

    parts = meta_member_path.split("/")
    if len(parts) >= 2 and parts[0] == "runs":
        return parts[1]

    return zip_name


# -----------------------------
# Public APIs
# -----------------------------

def load_trial_from_zip_bytes(zip_name: str, zip_bytes: bytes) -> TrialRecord:
    """
    アップロードZIP(1件)から TrialRecord を復元。
    必須：meta.json, ir_end.json, chat_log.jsonl
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        meta_member = _find_member(zf, "meta.json")
        ir_member = _find_member(zf, "ir_end.json") or _find_member(zf, "ir.json")
        log_member = _find_member(zf, "chat_log.jsonl") or _find_member(zf, "log.jsonl")

        missing = []
        if not meta_member:
            missing.append("meta.json")
        if not ir_member:
            missing.append("ir_end.json (or ir.json)")
        if not log_member:
            missing.append("chat_log.jsonl (or log.jsonl)")
        if missing:
            raise ValueError(f"{zip_name}: ZIP内に必要ファイルが不足: {', '.join(missing)}")

        meta = _read_json(zf, meta_member)
        ir = _read_json(zf, ir_member)

        lines = _read_text_lines(zf, log_member)
        chat_rows = _parse_chat_jsonl_lines(lines)
        qa_pairs = _build_qa_pairs(chat_rows)

        run_id = _infer_run_id(meta_member, meta, zip_name)

        return TrialRecord(
            zip_name=zip_name,
            run_id=run_id,
            meta=meta,
            ir=ir,
            chat_rows=chat_rows,
            qa_pairs=qa_pairs,
        )


def load_trials_from_uploaded_files(uploaded_files) -> List[TrialRecord]:
    """
    Streamlit の st.file_uploader(accept_multiple_files=True) の返り値を想定。
    """
    trials: List[TrialRecord] = []
    for uf in uploaded_files:
        trials.append(load_trial_from_zip_bytes(uf.name, uf.getvalue()))
    return trials

