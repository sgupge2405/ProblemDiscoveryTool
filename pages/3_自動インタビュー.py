from __future__ import annotations

import streamlit as st
import re
import json
import io
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Tuple

from chains import run_evidence_attach, run_persona_answer, run_evidence_finalize
from storage import load_ir_history_for_session
from admin_rag import get_admin_hints
from export_bundle import (
    build_run_zip,
    make_run_id,
    build_chat_log_jsonl,
    build_chat_log_md,
)

# -------------------------------------
# Streamlit 基本設定
#   ※ マルチページ構成なら set_page_config は app.py のみで行う
# -------------------------------------
# st.set_page_config(page_title="自動インタビュー", layout="centered")
st.title("自動インタビュー")
st.caption("検証用（AI×ペルソナ）: 終了時にログ(md/jsonl)とIR(evidence)をZIPでダウンロードできます。")

# -------------------------------------
# セッション状態の初期化（この後のUI/実行で使う）
# -------------------------------------
# 選択中 Session ID / 第1モジュールIR / objective_card
if "auto_selected_sid" not in st.session_state:
    st.session_state["auto_selected_sid"] = ""
if "auto_base_ir" not in st.session_state:
    st.session_state["auto_base_ir"] = None
if "auto_oc" not in st.session_state:
    st.session_state["auto_oc"] = None

# 対話履歴（Q,A）: List[Tuple[str,str]]
if "auto_history" not in st.session_state:
    st.session_state["auto_history"] = []

# 停止フラグ
if "auto_stop" not in st.session_state:
    st.session_state["auto_stop"] = False
if "auto_stop_reason" not in st.session_state:
    st.session_state["auto_stop_reason"] = ""

# 終了後の生成物（ZIP/ログ等）
if "auto_ir_end" not in st.session_state:
    st.session_state["auto_ir_end"] = None
if "auto_log_jsonl" not in st.session_state:
    st.session_state["auto_log_jsonl"] = ""
if "auto_log_md" not in st.session_state:
    st.session_state["auto_log_md"] = ""
if "auto_run_id" not in st.session_state:
    st.session_state["auto_run_id"] = ""
if "auto_zip_bytes" not in st.session_state:
    st.session_state["auto_zip_bytes"] = None
if "auto_zip_name" not in st.session_state:
    st.session_state["auto_zip_name"] = ""

if "auto_trial" not in st.session_state:
    st.session_state["auto_trial"] = 1


# -------------------------------------
# URLクエリ (?sid=...) を読む（保持だけ）
#   ※ load_base_ir_and_oc を呼ぶのは「関数定義の後」に行う
# -------------------------------------
sid_from_url = st.query_params.get("sid")
if sid_from_url and not st.session_state.get("auto_selected_sid"):
    st.session_state["auto_selected_sid"] = sid_from_url

# -------------------------------------
# 改行整形関数（UI見た目だけ整える）
# -------------------------------------
def format_for_chat(text: str) -> str:
    """見やすいように中点と改行を軽く整形"""
    text = re.sub(r"^\s*・", "\n\n・", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -------------------------------------
# 終了トークン（質問側が @@ を出したら終了）
# -------------------------------------
STOP_TOKENS = {"@@", "＠＠"}

def is_stop(s: str) -> bool:
    return re.sub(r"\s+", " ", s).strip() in STOP_TOKENS

# -------------------------------------
# LLM出力から「質問文」だけを抽出
# -------------------------------------
def extract_question(out: Dict[str, Any]) -> str:
    for k in ("question", "next_question", "q", "text", "content"):
        v = out.get(k)
        if isinstance(v, str) and v.strip():
            candidate = v.strip()
            break
    else:
        return ""

    lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
    if not lines:
        return ""

    # 明示ラベル（質問: / Q:）を優先
    for ln in lines:
        m = re.match(r"^(質問|Q)\s*[:：]\s*(.+)$", ln)
        if m:
            return m.group(2).strip()

    # 疑問符で終わる行を優先
    for ln in lines:
        if ln.endswith("？") or ln.endswith("?"):
            return ln

    # 最後の手段：先頭行
    return lines[0]

# -------------------------------------
# 保存済みIRから「第1モジュールIR」と objective_card を復元
# -------------------------------------
def load_base_ir_and_oc(session_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    items = load_ir_history_for_session(session_id, limit=200)
    cand: Dict[str, Any] | None = None

    # 取得順が不確実でも「最新の extraction」を拾えるよう、後ろから探索
    for row in reversed(items):
        ir = row.get("ir")
        if isinstance(ir, dict) and ir.get("stage") == "extraction":
            cand = ir
            break

    # 見つからなければ、最新のdictを採用（後ろから）
    if cand is None:
        for row in reversed(items):
            ir = row.get("ir")
            if isinstance(ir, dict):
                cand = ir
                break

    if cand is None:
        raise RuntimeError("IR が見つかりませんでした。session_id を確認してください。")

    oc = cand.get("objective_card")
    if not isinstance(oc, dict):
        stage = cand.get("stage")
        raise RuntimeError(
            f"objective_card が見つかりませんでした（stage={stage}）。IR形式を確認してください。"
        )

    return cand, oc

# -------------------------------------
# extras を安全に正規化（list[str] に寄せる）
# -------------------------------------
def normalize_extras(oc: Dict[str, Any]) -> List[str]:
    extras_raw = oc.get("extras") or []
    if isinstance(extras_raw, list):
        return [str(x) for x in extras_raw if str(x).strip()]
    if str(extras_raw).strip():
        return [str(extras_raw)]
    return []

# -------------------------------------
# 質問側（インタビュアー）向け文脈
# -------------------------------------
def build_context_for_interviewer(oc: Dict[str, Any], history: List[Tuple[str, str]]) -> str:
    extras = normalize_extras(oc)

    hist_text = ""
    if history:
        recent = history[-10:]
        chunks: List[str] = []
        start_idx = max(1, len(history) - len(recent) + 1)
        for i, (q, a) in enumerate(recent, start=start_idx):
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
        "次に尋ねるべき質問を1つだけ生成してください（質問文1行のみ）。"
        " もう十分に深掘りでき、これ以上質問が不要だと判断した場合は @@ のみを出力してください（@@以外は付けない）。\n"
    )

# -------------------------------------
# 回答側（ペルソナ）向け文脈
# -------------------------------------
def build_context_for_persona(oc: Dict[str, Any], question: str) -> str:
    extras = normalize_extras(oc)

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

# -------------------------------------
# revision 採番（既存最大 + 1）
#   ※ evidence のみで採番（評価で扱いやすくする）
# -------------------------------------
def next_revision_for_session(session_id: str) -> int:
    items = load_ir_history_for_session(session_id, limit=500)
    revs: List[int] = []
    for row in items:
        ir = row.get("ir")
        if (
            isinstance(ir, dict)
            and ir.get("stage") == "evidence"
            and isinstance(ir.get("revision"), int)
        ):
            revs.append(ir["revision"])
    return (max(revs) + 1) if revs else 1

# -------------------------------------
# 確定生成（explanation追加）用にログを整形
# -------------------------------------
def build_history_text_for_finalize(history: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for i, (q, a) in enumerate(history, start=1):
        lines.append(f"[{i}] インタビュアー：{q}")
        lines.append(f"    ユーザー：{a}")
    return "\n".join(lines) if lines else "（まだ会話なし）"

# -------------------------------------
# 終了時：IR(evidence) を確定生成（PCダウンロード用）
# -------------------------------------
def finalize_and_build_ir_end() -> Dict[str, Any]:
    """終了時に1回だけ IR(evidence) を確定生成して返す（rerun対策あり）。"""
    cached = st.session_state.get("auto_ir_end")
    if isinstance(cached, dict):
        return cached

    sid = st.session_state.get("auto_selected_sid") or ""
    if not sid:
        raise RuntimeError("session_id が未設定です。先にIRを読み込んでください。")

    base_ir = st.session_state.get("auto_base_ir")
    history = st.session_state.get("auto_history", [])

    if not isinstance(base_ir, dict):
        raise RuntimeError("base_ir がありません（IR読み込みをやり直してください）")

    history_text = build_history_text_for_finalize(history)
    base_ir_json = json.dumps(base_ir, ensure_ascii=False, indent=2)

    finalize_input = (
        "【第1モジュールIR（objective_card を含む）】\n"
        f"{base_ir_json}\n\n"
        "【根拠付与インタビューの全ログ】\n"
        f"{history_text}\n\n"
        "【指示】\n"
        "上のログを根拠として，objective_card の各項目に対応する explanation を追記し，"
        "更新後の IR(JSON) を1つだけ出力してください．\n"
        "- 出力は JSON のみ（前後説明禁止）\n"
        "- target_user/problem/contradiction/objective/value の各 *_explanation は配列で，"
        "  今回追加する要約（根拠）を必要な分だけ append してください．\n"
    )

    fin = run_evidence_finalize(finalize_input)
    if fin.get("mode") != "IR" or not isinstance(fin.get("ir"), dict):
        raise RuntimeError("IR(JSON)の確定生成に失敗しました（mode!=IR）。")

    ir_end = fin["ir"]
    ir_end["session_id"] = sid
    ir_end["stage"] = "evidence"
    ir_end["revision"] = next_revision_for_session(sid)

    # 保存はしない：メモリに保持してZIPダウンロードへ
    st.session_state["auto_ir_end"] = ir_end
    return ir_end

# -------------------------------------
# 終了時：ログ＋IR を生成し、ZIPを用意（PCダウンロード用）
# -------------------------------------
def finalize_and_build_all(stop_reason: str) -> None:
    """
    auto(検証)の終了処理（PCダウンロード用）：
      1) IR(evidence) を確定生成（1回だけ）
      2) export_bundle でログ＋IR＋meta をZIP化（1回だけ）
      3) download_button 用に session_state に保持
    """
    st.session_state["auto_stop"] = True
    st.session_state["auto_stop_reason"] = stop_reason

    sid = st.session_state.get("auto_selected_sid") or ""
    if not sid:
        raise RuntimeError("session_id が未設定です。先にIRを読み込んでください。")

    # rerun対策：ZIPが既にあれば何もしない
    if st.session_state.get("auto_zip_bytes"):
        return

    history = st.session_state.get("auto_history", [])

    # 1) ir_end（dict）
    ir_end = finalize_and_build_ir_end()

    # 2) ZIP生成（export_bundleに完全移管）
    zip_bytes, run_id, zip_name = build_run_zip(
        history=history,
        session_id=sid,
        stop_reason=stop_reason,
        ir_end=ir_end,
        tool="app_evidence_interview_auto_0105",
        admin_rag=st.session_state.get("auto_use_admin_rag", False),
        run_mode=st.session_state.get("auto_run_mode", "step"),
    )

    st.session_state["auto_log_jsonl"] = ""
    st.session_state["auto_log_md"] = ""

    st.session_state["auto_run_id"] = run_id
    st.session_state["auto_zip_bytes"] = zip_bytes
    st.session_state["auto_zip_name"] = zip_name

# =========================================================
# ★追加：URLの ?sid= を受け取ったら自動でIRロード（初回のみ）
#   ※ load_base_ir_and_oc が定義済みの位置なので、ここで呼べる
# =========================================================
sid_from_url = st.query_params.get("sid")
if sid_from_url and not st.session_state.get("auto_selected_sid"):
    st.session_state["auto_selected_sid"] = sid_from_url

if sid_from_url and st.session_state.get("auto_oc") is None:
    try:
        base_ir, oc = load_base_ir_and_oc(sid_from_url)
    except Exception as e:
        st.error(f"URLのsidからIR読み込みに失敗しました: {e}")
    else:
        st.session_state["auto_selected_sid"] = sid_from_url
        st.session_state["auto_base_ir"] = base_ir
        st.session_state["auto_oc"] = oc

        # 会話状態を新規スタート
        st.session_state["auto_history"] = []
        st.session_state["auto_stop"] = False
        st.session_state["auto_stop_reason"] = ""

        # 生成物もクリア（rerun対策キー）
        st.session_state["auto_ir_end"] = None
        st.session_state["auto_log_jsonl"] = ""
        st.session_state["auto_log_md"] = ""
        st.session_state["auto_run_id"] = ""
        st.session_state["auto_zip_bytes"] = None
        st.session_state["auto_zip_name"] = ""

        st.toast("Session ID を受け取り、IRを自動ロードしました。", icon="✅")

# -------------------------------------
# sidebar: Session ID → IR 読み込み / リセット
#  - reset/load で「会話状態」と「生成物」を正しく初期化する
# -------------------------------------
with st.sidebar:
    st.subheader("対象となる Session ID を指定")

    sid_input = st.text_input(
        "第1モジュールで保存された Session ID",
        value=st.session_state.get("auto_selected_sid", ""),
        placeholder="例: 123e4567-e89b-12d3-a456-426614174000",
    )

    st.session_state["auto_max_turns"] = st.number_input(
        "最大ターン数（安全装置）",
        min_value=1,
        max_value=50,
        value=int(st.session_state.get("auto_max_turns", 20)),
        step=1,
    )

    # ---------------------------------
    # ボタン群（※リセットを先に処理するため、ウィジェットより前に置く）
    # ---------------------------------
    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("IR を読み込む", use_container_width=True)
    with col2:
        reset_btn = st.button("リセット", use_container_width=True)

    # --- リセット（新規スタート状態へ） ---
    if reset_btn:
        # 入力/文脈
        st.session_state["auto_selected_sid"] = ""
        st.session_state["auto_oc"] = None
        st.session_state["auto_base_ir"] = None

        # 会話
        st.session_state["auto_history"] = []
        st.session_state["auto_stop"] = False
        st.session_state["auto_stop_reason"] = ""

        # 生成物（PCダウンロード用）を全消し
        st.session_state["auto_ir_end"] = None
        st.session_state["auto_log_jsonl"] = ""
        st.session_state["auto_log_md"] = ""
        st.session_state["auto_run_id"] = ""
        st.session_state["auto_zip_bytes"] = None
        st.session_state["auto_zip_name"] = ""

        # 実行オプションを初期化（※ウィジェット生成前なのでOK）
        st.session_state["auto_use_admin_rag"] = False
        st.session_state["auto_run_mode"] = "step"

        # URLパラメータも消す
        try:
            st.query_params.clear()
        except Exception:
            pass

        st.rerun()

    # ---------------------------------
    # 実行オプション（管理者RAG / 処理方式）
    #   ※ key を付けて session_state を Streamlit に任せる
    # ---------------------------------
    st.divider()
    st.subheader("実行オプション")

    st.checkbox(
        "管理者RAGを使う（質問生成のヒントを内部付与）",
        key="auto_use_admin_rag",
        value=st.session_state.get("auto_use_admin_rag", False),
    )

    st.radio(
        "処理方式",
        options=["step", "batch"],
        format_func=lambda x: "逐次（1ターンずつ）" if x == "step" else "一括（停止まで自動）",
        key="auto_run_mode",
        index=0 if st.session_state.get("auto_run_mode", "step") == "step" else 1,
    )

    # --- IR 読み込み ---
    if load_btn:
        sid = (sid_input or "").strip()
        if not sid:
            st.error("Session ID を入力してください。")
        else:
            try:
                base_ir, oc = load_base_ir_and_oc(sid)
            except Exception as e:
                st.error(f"IR読み込みに失敗しました: {e}")
            else:
                st.session_state["auto_selected_sid"] = sid
                st.session_state["auto_base_ir"] = base_ir
                st.session_state["auto_oc"] = oc

                st.session_state["auto_history"] = []
                st.session_state["auto_stop"] = False
                st.session_state["auto_stop_reason"] = ""

                st.session_state["auto_ir_end"] = None
                st.session_state["auto_log_jsonl"] = ""
                st.session_state["auto_log_md"] = ""
                st.session_state["auto_run_id"] = ""
                st.session_state["auto_zip_bytes"] = None
                st.session_state["auto_zip_name"] = ""

                try:
                    st.query_params["sid"] = sid
                except Exception:
                    pass

                st.success("IR を読み込みました。")
                st.rerun()

# -------------------------------------
# main
# -------------------------------------
oc = st.session_state.get("auto_oc")
if oc is None:
    st.info("左のサイドバーで Session ID を指定し、IR を読み込んでください。")
    st.stop()

st.markdown("### 文脈")
st.markdown(f"- **対象ユーザー**：{oc.get('target_user', '')}")
st.markdown(f"- **課題**：{oc.get('problem', '')}")
st.markdown(f"- **矛盾**：{oc.get('contradiction', '')}")
st.markdown(f"- **目的**：{oc.get('objective', '')}")
st.markdown(f"- **価値**：{oc.get('value', '')}")
st.markdown(f"- **背景**：{oc.get('background', '')}")

extras = normalize_extras(oc)
if extras:
    st.markdown("**補足情報**：")
    for ex in extras:
        st.markdown(f"- {ex}")

st.markdown("---")

history: List[Tuple[str, str]] = st.session_state.get("auto_history", [])
max_turns = int(st.session_state.get("auto_max_turns", 20))

# -------------------------------------
# チャット表示（インタビュアー=assistant、ペルソナ=user）
# -------------------------------------
for i, (q, a) in enumerate(history, start=1):
    with st.chat_message("assistant"):
        st.markdown(f"**[Turn {i}] 質問**\n\n{q}")
    with st.chat_message("user"):
        st.markdown(f"**[Turn {i}] 回答**\n\n{format_for_chat(a)}")

# -------------------------------------
# 終了表示（ZIPダウンロード）
# ※ 終了処理は finalize_and_build_all() に集約されている前提
# -------------------------------------
if st.session_state.get("auto_stop"):
    st.success("終了しました。評価実験用データ（ログ＋IR＋メタ）をZIPでダウンロードしてください。")

    # ★追加：同一sidで再試行（会話だけリセット）
    if st.button("同じSession IDでもう一度（新しい試行を開始）", use_container_width=True):
        # ★ trial を進める
        st.session_state["auto_trial"] += 1
        # 会話を初期化
        st.session_state["auto_history"] = []
        st.session_state["auto_stop"] = False
        st.session_state["auto_stop_reason"] = ""
        
        # 生成物を初期化
        st.session_state["auto_ir_end"] = None
        st.session_state["auto_log_jsonl"] = ""
        st.session_state["auto_log_md"] = ""
        st.session_state["auto_run_id"] = ""
        st.session_state["auto_zip_bytes"] = None
        st.session_state["auto_zip_name"] = ""
        
        st.rerun()

    run_id = st.session_state.get("auto_run_id", "")
    if run_id:
        st.markdown(f"- run_id: `{run_id}`")
    
    # ★ trial は run_id が無くても表示する
    st.markdown(f"- trial: `{st.session_state.get('auto_trial', 1)}`")


    ir_end = st.session_state.get("auto_ir_end")
    if isinstance(ir_end, dict) and isinstance(ir_end.get("revision"), int):
        st.markdown(f"- ir_end.revision: `{ir_end['revision']}`")

    st.warning(f"終了：{st.session_state.get('auto_stop_reason', '')}")

    zip_bytes = st.session_state.get("auto_zip_bytes")
    zip_name = st.session_state.get("auto_zip_name", "run.zip")
    if zip_bytes:
        st.download_button(
            label="実験データをZIPでダウンロード（chat_log + ir_end + meta）",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.error("ZIP生成データがありません（auto_zip_bytes が空です）。")

    st.stop()

# -------------------------------------
# 操作ボタン
#  - 生成：1ターン進める（step） or 停止まで回す（batch）
#  - 途中ログDL：現時点のログをZIPでダウンロード（任意）
# -------------------------------------
colA, colB = st.columns([1, 1])
with colA:
    gen_btn = st.button("次のターンを生成", use_container_width=True)
with colB:
    save_btn = st.button("途中ログをダウンロード（jsonl + md）", use_container_width=True)

def apply_admin_rag_if_enabled(ctx: str) -> str:
    """質問生成の文脈にだけ管理者RAGを内部付与する（UI表示はしない）。"""
    if not st.session_state.get("auto_use_admin_rag", False):
        return ctx
    try:
        hints = get_admin_hints(ctx)
    except Exception:
        hints = ""
    if hints:
        ctx += (
            "\n\n【内部向けヒント（管理者RAG）】\n"
            "以下は質問観点・テンプレートのヒントです。ユーザーへの表示は禁止。\n"
            f"{hints}"
        )
    return ctx

def run_one_turn() -> str:
    """
    1ターンだけ進める。
    return: "" なら継続、非空文字なら stop_reason（終了）
    """
    # --- ガード：IR未ロードなら実行しない ---
    if st.session_state.get("auto_oc") is None or st.session_state.get("auto_base_ir") is None:
        return "not_ready(no_ir_loaded)"

    history = st.session_state.get("auto_history", [])

    # max_turns 到達 → 終了（安全装置）
    if len(history) >= max_turns:
        return f"max_turns({max_turns})"

    interviewer_ctx = build_context_for_interviewer(oc, history)
    interviewer_ctx = apply_admin_rag_if_enabled(interviewer_ctx)

    try:
        q_out = run_evidence_attach(interviewer_ctx)
    except Exception as e:
        return f"error(run_evidence_attach:{e})"

    question = extract_question(q_out) or "（質問の生成に失敗しました。どの点を深掘りすべきですか？）"

    # @@ → 終了
    if is_stop(question):
        return "interviewer_stop(@@)"

    persona_ctx = build_context_for_persona(oc, question)
    try:
        a_out = run_persona_answer(persona_ctx)
    except Exception as e:
        return f"error(run_persona_answer:{e})"

    answer = (a_out.get("text") or "").strip()
    if not answer:
        answer = "（回答の生成に失敗しました。前提情報をもう少し具体化してください。）"

    history.append((question, answer))
    st.session_state["auto_history"] = history
    return ""

# -------------------------------------
# 生成ボタン押下時：step / batch を切り替え
# -------------------------------------
if gen_btn:
    # --- ガード：IR未ロードなら案内して止める ---
    if st.session_state.get("auto_oc") is None:
        st.info("先に左のサイドバーで Session ID を指定し、IR を読み込んでください。")
        st.stop()

    mode = st.session_state.get("auto_run_mode", "step")

    # ---- 一括（停止まで自動） ----
    if mode == "batch":
        # 暴走防止：最後に増えたターン数をカウントし、上限で止める
        # （max_turns で止まる想定だが、保険として）
        guard_limit = max_turns + 2

        with st.spinner("一括実行中（停止条件まで自動生成）..."):
            start_len = len(st.session_state.get("auto_history", []))

            while True:
                stop_reason = run_one_turn()

                if stop_reason:
                    stop_reason = f"{stop_reason};trial={st.session_state.get('auto_trial', 1)}"
                    finalize_and_build_all(stop_reason)
                    break

                # 念のための二重安全装置（予期せぬループ対策）
                now_len = len(st.session_state.get("auto_history", []))
                if (now_len - start_len) >= guard_limit:
                    finalize_and_build_all(f"guard_loop(limit={guard_limit})")
                    break

        st.rerun()

    # ---- 逐次（1ターンずつ） ----
    else:
        with st.spinner("1ターン生成しています..."):
            stop_reason = run_one_turn()

        if stop_reason:
            with st.spinner("終了処理：ログ＋IR＋メタをZIP化しています..."):
                stop_reason = f"{stop_reason};trial={st.session_state.get('auto_trial', 1)}"
                finalize_and_build_all(stop_reason)

        st.rerun()

# -------------------------------------
# 途中ログのダウンロード（任意）
#  - 自動終了前でも、現時点のログ（jsonl/md）をZIPで落とせる
#  - IR(evidence) はまだ確定しない（入れない）
# -------------------------------------
if save_btn:
    sid = st.session_state.get("auto_selected_sid") or ""
    if not sid:
        st.error("session_id が未設定です。先にIRを読み込んでください。")
    else:
        history_now = st.session_state.get("auto_history", [])

        jsonl_str = build_chat_log_jsonl(history_now, sid)
        md_str = build_chat_log_md(history_now, sid, stop_reason="in_progress")

        run_id = make_run_id(prefix="logs")
        buf = io.BytesIO()
        base = f"runs/{run_id}/"
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(base + "chat_log.jsonl", jsonl_str)
            z.writestr(base + "chat_log.md", md_str)

        # ★download_button を押すまで buf が生きるよう Bytes を確保
        zip_data = buf.getvalue()

        st.download_button(
            label="途中ログZIPをダウンロード",
            data=zip_data,
            file_name=f"{run_id}.zip",
            mime="application/zip",
            use_container_width=True,
        )