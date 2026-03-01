# app_problem_extract.py — 第1モジュール（問題意識抽出）UI
import streamlit as st
from datetime import datetime
import re
from io import BytesIO  # ← docx読み込みで使う

from chains import run_problem_extract
from storage import save_ir_jsonl, load_ir_history_for_session
# 既存の import 群のところに追加
from admin_rag import get_admin_hints

# ← docx, pdf を「入っていたら使う」方式で読む
try:
    import docx
except ImportError:
    docx = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

st.set_page_config(page_title="問題意識抽出（第1モジュール）", layout="centered")
st.title("問題意識抽出")

# -------------------------------------
# 改行整形関数（UI見た目だけ整える）
# -------------------------------------
def format_for_chat(text: str) -> str:
    """見やすいように中点と改行を軽く整形"""
    text = re.sub(r'^\s*・', '\n\n・', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# -------------------------------------
# 確定入力判定（@@ / ＠＠）
# -------------------------------------
CONFIRM_PAT = re.compile(r"^(@@|＠＠)[！!。\.]?$", re.IGNORECASE)

def is_confirmation_or_brief_addition(text: str) -> bool:
    """承認っぽい短文 or 追記が短い場合を検知"""
    t = text.strip().replace("　", "")
    # 全角＠を半角に寄せる
    t = t.replace("＠", "@")
    if CONFIRM_PAT.fullmatch(t):
        return True
    # 追記が短文なら続き扱い
    return len(t) <= 100

def build_ir_request(pending_summary: str, user_text: str) -> str:
    """前回要約＋今回の承認/追記を合成してIR確定を指示"""
    return (
        "【前回の要約】\n"
        f"{pending_summary}\n\n"
        "【承認/追記】\n"
        f"{user_text}\n\n"
        "【指示】\n"
        "上記を前提に，正式なIR(JSON)を1つだけ出力してください．"
        "今回はJSONのみを返し，他の説明文は出さないでください．"
    )

def is_pure_confirmation(text: str) -> bool:
    """@@（または＠＠）だけの確定入力を検知"""
    t = text.strip().replace("　", "")
    t = t.replace("＠", "@")
    return CONFIRM_PAT.fullmatch(t) is not None

# -------------------------------
# セッション状態の初期化
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_sid" not in st.session_state:
    st.session_state["last_sid"] = ""
if "pending_summary" not in st.session_state:
    st.session_state["pending_summary"] = ""
if "continue_mode" not in st.session_state:
    st.session_state["continue_mode"] = "auto"
# ← アップロードしたテキストを一時保持する場所
if "uploaded_context" not in st.session_state:
    st.session_state["uploaded_context"] = ""


# -------------------------------
# サイドバー：アップロード＋履歴
# -------------------------------
with st.sidebar:
    st.subheader("参考資料をアップロード")
    up_file = st.file_uploader("txt / docx / pdf に対応", type=["txt", "docx", "pdf"])
    if up_file is not None:
        text_from_file = ""
        fname = up_file.name.lower()

        if fname.endswith(".txt"):
            text_from_file = up_file.read().decode("utf-8", errors="ignore")

        elif fname.endswith(".docx"):
            if docx is None:
                st.warning("python-docx が無いので docx を読めません")
            else:
                doc = docx.Document(BytesIO(up_file.read()))
                text_from_file = "\n".join(p.text for p in doc.paragraphs)

        elif fname.endswith(".pdf"):
            if fitz is None:
                st.warning("PyMuPDF (fitz) が無いので pdf を読めません")
            else:
                with fitz.open(stream=up_file.read(), filetype="pdf") as pdf_doc:
                    pages = [page.get_text("text") for page in pdf_doc]
                    text_from_file = "\n".join(pages)

        # 長すぎるとLLMに入りきらないので頭だけ
        st.session_state["uploaded_context"] = text_from_file[:3000]
        st.success("参考資料を読み込みました（先頭3000文字を使用）")

    st.markdown("---")

    st.subheader("履歴（同一 Session ID）")
    sid_query = st.text_input("Session ID を指定", value=st.session_state.get("last_sid", ""))
    if sid_query:
        items = load_ir_history_for_session(sid_query, limit=20)
        st.caption(f"一致 {len(items)} 件（新しい順）")
        for i, row in enumerate(items, 1):
            with st.expander(f"{i}. {row['ts']} / mode={row['mode']}"):
                st.json(row.get("ir") or {"text": row.get("text", "")})

    st.markdown("---")
    st.subheader("続き判定モード")
    st.radio(
        "要約後の短文を自動で“確定”とみなすか",
        options=["auto", "force_on", "force_off"],
        key="continue_mode",
        format_func=lambda x: {
            "auto": "自動（@@, ＠＠ か短文なら続き扱い）",
            "force_on": "常に続き扱いにする",
            "force_off": "常に新規扱いにする",
        }[x],
    )

    # ★ ここから下を追加 ★
    st.markdown("---")
    st.subheader("管理者RAGの利用")
    st.checkbox(
        "管理者ファイル（講義資料など）を内部ヒントとして使う",
        key="use_admin_rag",
        value=True,  # デフォルトON（評価実験で OFF にもできる）
    )

# -------------------------------
# これまでの会話表示
# -------------------------------
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and m.get("mode") == "IR":
            ir_obj = m["ir"]
            oc = ir_obj.get("objective_card") or {}
            st.markdown(f"**目的**：{oc.get('objective','')}")
            st.markdown(f"**課題**：{oc.get('problem','')}")
            st.markdown(f"**矛盾**：{oc.get('contradiction','')}")
            st.markdown(f"**価値**：{oc.get('value','')}")
            st.markdown(f"**背景**：{oc.get('background','')}")
            extras = oc.get("extras") or []
            if extras:
                st.markdown("**補足**：")
                for ex in extras:
                    st.markdown(f"- {ex}")
            with st.expander("内部IRを表示（開発者向け）"):
                st.json(m["ir"])
            st.caption(f"Session ID: `{m.get('sid','')}` / {m['ts']}")
        else:
            st.markdown(m["content"], unsafe_allow_html=False)
            sid_show = m.get("sid")
            if sid_show:
                st.caption(f"Session ID: `{sid_show}` / {m['ts']}")

# -------------------------------
# 入力受付
# -------------------------------
user_text = st.chat_input("自由発話またはファイルを送ってください")

if user_text:
    ts = datetime.utcnow().isoformat() + "Z"

    # まずユーザーの発話を表示
    with st.chat_message("user"):
        st.markdown(user_text, unsafe_allow_html=False)

    st.session_state["messages"].append(
        {"role": "user", "content": user_text, "mode": "USER", "sid": None, "ts": ts}
    )

    # --- ここからが「どこに足すか」の本題です ---

    # 1) いままで通り「続き扱い」かどうかを先に判定 + @@は確定扱い
    cont_mode = st.session_state["continue_mode"]
    pending = st.session_state["pending_summary"].strip()
    pure_confirm = is_pure_confirmation(user_text)  # ← 追加

    if cont_mode == "force_on":
        treat_as_continuation = bool(pending)
    elif cont_mode == "force_off":
        treat_as_continuation = False
    else:  # auto
        # @@（純確定）なら続き扱いに含める
        treat_as_continuation = bool(pending) and (pure_confirm or is_confirmation_or_brief_addition(user_text))

    # 2) ここで LLM に渡す“元の文字列”を作る
    if treat_as_continuation:
        # 純粋な確定（@@）なら、追記テキストは固定で "@@" を渡す
        ack_text = "@@" if pure_confirm else user_text
        composed_input = build_ir_request(pending, ack_text)
    else:
        composed_input = user_text

    # 3) そしてここで「アップロード分をくっつける」
    extra = st.session_state.get("uploaded_context") or ""
    if extra:
        composed_input = (
            f"{composed_input}\n\n"
            "【参考資料（アップロード分）】\n"
            "以下は利用者が同時に提出した資料です。今回の目的・課題・矛盾・価値を具体化するためにのみ参照してください。\n"
            f"{extra}"
        )

    # 4) @@ の場合は「必ずIR(JSON)のみ」を厳格指示で追加（強制）
    if pure_confirm:
        composed_input += (
            "\n\n【厳格指示】"
            "今から必ず IR(JSON) を1件のみ返してください。"
            "途中説明・確認文は一切返さないこと。"
        )

    # ★ 4.5) 管理者RAGからのヒントを付与（必要な場合のみ）★
    if st.session_state.get("use_admin_rag", False):
        admin_hints = get_admin_hints(user_text)
        if admin_hints:
            composed_input += (
                "\n\n【内部向けヒント（管理者ドキュメント由来）】\n"
                "以下は開発者が用意した管理者ドキュメントの抜粋です。ユーザーにはこの文面を直接見せず、"
                "質問の観点や整理の仕方の参考としてのみ利用してください。"
                "授業名・資料名・ページ番号などの固有情報は、ユーザーへの回答に含めないでください。\n"
                f"{admin_hints}"
            )

    # 5) あとはいつも通り解析して表示
    with st.chat_message("assistant"):
        with st.spinner("抽出中..."):
            try:
                result = run_problem_extract(composed_input)
                # st.write("🔥 debug result:", result)  # ← デバッグ終わったらコメントアウト/削除OK
            except Exception as e:
                st.error(f"抽出処理でエラーが発生しました: {e}")
                st.session_state["messages"].append(
                    {"role": "assistant", "content": f"抽出エラー: {e}", "mode": "ERROR", "sid": None, "ts": ts}
                )
                st.stop()

        mode = result.get("mode")

        # フェイルセーフ：@@ なのに IR で返らない場合は、より強い指示でワンモアトライ
        if pure_confirm and mode != "IR":
            strict_req = build_ir_request(pending, "@@") + (
                "\n\n【厳格指示】"
                "今から必ず IR(JSON) を1件のみ返してください。"
                "途中説明・確認文は一切返さないこと。"
            )
            try:
                result = run_problem_extract(strict_req)
                mode = result.get("mode")
            except Exception:
                pass

        if mode == "ASK_OR_SUMMARY":
            # "message" がなければ "text" を使う（モデル差異に強くする）
            raw_msg = result.get("message") or result.get("text") or ""
            text = format_for_chat(raw_msg)
            st.markdown(text, unsafe_allow_html=False)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": text,
                    "mode": "ASK_OR_SUMMARY",
                    "sid": None,
                    "ts": ts,
                }
            )
            st.session_state["pending_summary"] = text

        elif mode == "IR":
            ir_obj = result.get("ir", {})
            ir_obj.setdefault("revision", 1)
            ir_obj.setdefault("stage", "extraction")

            extra = st.session_state.get("uploaded_context") or ""
            if extra:
                ir_obj.setdefault("refs", [])
                ir_obj["refs"].append({"type": "upload", "note": "user uploaded file used in extraction"})

            sid = ir_obj.get("session_id", "(no id)")

            oc = ir_obj.get("objective_card") or {}
            st.markdown(f"**目的**：{oc.get('objective','')}")
            st.markdown(f"**課題**：{oc.get('problem','')}")
            st.markdown(f"**矛盾**：{oc.get('contradiction','')}")
            st.markdown(f"**価値**：{oc.get('value','')}")
            st.markdown(f"**背景**：{oc.get('background','')}")
            extras = oc.get("extras") or []
            if extras:
                st.markdown("**補足**：")
                for ex in extras:
                    st.markdown(f"- {ex}")

            with st.expander("内部IRを表示（開発者向け）"):
                st.json(ir_obj)

            try:
                save_ir_jsonl(ir_obj)
            except Exception as e:
                st.error(f"IRの保存に失敗しました: {e}")

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": "IR を生成しました",
                    "mode": "IR",
                    "ir": ir_obj,
                    "sid": sid,
                    "ts": ts,
                }
            )
            st.session_state["last_sid"] = sid
            st.session_state["pending_summary"] = ""
            # --- ここから追加：保存直後の導線（SIDコピー＋次工程へ） ---
            st.markdown("---")
            st.subheader("次の工程へ")
            
            st.markdown("### Session ID（コピー）")
            st.text_input("Session ID", value=sid, key=f"sid_copy_{sid}")
            
            colA, colB = st.columns(2)
            
            with colA:
                if st.button("根拠付与へ進む", use_container_width=True, key=f"to_evidence_{sid}"):
                    # 次ページで自動入力できるようにクエリへ載せる
                    st.query_params["sid"] = sid
                    st.switch_page("pages/2_根拠付与.py")
            
            with colB:
                if st.button("自動インタビューへ進む", use_container_width=True, key=f"to_auto_{sid}"):
                    st.query_params["sid"] = sid
                    st.switch_page("pages/3_自動インタビュー.py")
                    
            st.caption("※うまく遷移しない場合でも、上のSession IDをコピーして各ページで貼り付ければOKです。")

        else:
            st.warning(f"未対応モードです: {mode}")
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": f"未対応モード: {mode}",
                    "mode": "WARN",
                    "sid": None,
                    "ts": ts,
                }
            )
