from __future__ import annotations
import streamlit as st
from datetime import datetime
import re

from chains import run_persona_answer
from storage import load_ir_history_for_session

st.set_page_config(page_title="インタビュー受け手（ペルソナ）", layout="centered")
st.title("インタビュー受け手エージェント（ペルソナ）")

# -------------------------------------
# 改行整形関数（UI見た目だけ整える）
# -------------------------------------
def format_for_chat(text: str) -> str:
    """見やすいように中点と改行を軽く整形"""
    text = re.sub(r'^\s*・', '\n\n・', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# -------------------------------------
# session state
# -------------------------------------
if "per_messages" not in st.session_state:
    st.session_state["per_messages"] = []
if "per_selected_sid" not in st.session_state:
    st.session_state["per_selected_sid"] = ""
if "per_selected_ir" not in st.session_state:
    st.session_state["per_selected_ir"] = None

# -------------------------------------
# sidebar: Session ID → IR 読み込み
# -------------------------------------
with st.sidebar:
    st.subheader("対象となる Session ID を指定")

    sid_input = st.text_input(
        "第1モジュールで保存された Session ID",
        value=st.session_state.get("per_selected_sid", ""),
        placeholder="例: 123e4567-e89b-12d3-a456-426614174000",
    )

    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("IR を読み込む", use_container_width=True)
    with col2:
        clear_btn = st.button("クリア", use_container_width=True)

    if clear_btn:
        st.session_state["per_selected_sid"] = ""
        st.session_state["per_selected_ir"] = None
        st.session_state["per_messages"] = []

    if load_btn and sid_input:
        items = load_ir_history_for_session(sid_input, limit=20)
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
            st.warning("指定した Session ID に対応する IR が見つかりませんでした。")
        else:
            st.session_state["per_selected_sid"] = sid_input
            st.session_state["per_selected_ir"] = candidate
            st.session_state["per_messages"] = []
            st.success("IR を読み込みました。")

# -------------------------------------
# main: 文脈表示
# -------------------------------------
selected_ir = st.session_state.get("per_selected_ir")
if selected_ir is None:
    st.info("左のサイドバーで Session ID を指定し、第1モジュールの IR を読み込んでください。")
    st.stop()

oc = selected_ir.get("objective_card", {})

st.markdown("### 第1モジュールで整理された内容（文脈）")
st.markdown(f"- **対象ユーザー**：{oc.get('target_user', '')}")
st.markdown(f"- **課題**：{oc.get('problem', '')}")
st.markdown(f"- **矛盾**：{oc.get('contradiction', '')}")
st.markdown(f"- **目的**：{oc.get('objective', '')}")
st.markdown(f"- **価値**：{oc.get('value', '')}")
st.markdown(f"- **背景**：{oc.get('background', '')}")

extras = oc.get("extras") or []
if extras:
    st.markdown("**補足情報**：")
    for ex in extras:
        st.markdown(f"- {ex}")

st.markdown("---")
st.markdown("ここからは、インタビュアーの質問に対して **対象ユーザー本人として** 回答します。")

# -------------------------------------
# chat log
# -------------------------------------
for m in st.session_state["per_messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------------------
# input
# -------------------------------------
q = st.chat_input("インタビュアーの質問を入力してください")
if q:
    now = datetime.utcnow().isoformat() + "Z"
    st.session_state["per_messages"].append(
        {"role": "user", "content": q, "ts": now}
    )

    # LLM に渡す文脈（内部語を含めない）
    context = (
        "【状況メモ】\n"
        f"対象ユーザー: {oc.get('target_user','')}\n"
        f"課題: {oc.get('problem','')}\n"
        f"矛盾: {oc.get('contradiction','')}\n"
        f"目的: {oc.get('objective','')}\n"
        f"価値: {oc.get('value','')}\n"
        f"背景: {oc.get('background','')}\n"
        f"補足: {', '.join(extras) if extras else ''}\n"
        "\n【質問】\n"
        f"{q}\n"
    )

    with st.spinner("回答を生成しています..."):
        try:
            out = run_persona_answer(context)
            ans = format_for_chat(out.get("text", ""))
        except Exception as e:
            ans = f"（エラーが発生しました）{e}"

    st.session_state["per_messages"].append(
        {"role": "assistant", "content": ans, "ts": now}
    )

    st.rerun()
