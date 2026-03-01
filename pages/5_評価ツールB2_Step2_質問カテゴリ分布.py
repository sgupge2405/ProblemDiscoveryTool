# pages/5_評価ツールB2_Step2_質問カテゴリ分布.py
"""評価実験ツールB2 Step2（質問カテゴリ分布）単体実行ページ

- run_*.zip（試行ZIP）をアップロード
- Step2（質問カテゴリ分類）だけを実行
- 結果（question_entropy_df / question_categories_df）を表示
- ダウンロードは step2_core_outputs.zip のみ

前提:
- evaluation/loader.py の load_trials_from_uploaded_files が利用できること
- evaluation/tool_b2_step2.py の run_tool_b2_step2_question_distribution が利用できること
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation.loader import load_trials_from_uploaded_files
from evaluation.tool_b2_step2 import run_tool_b2_step2_question_distribution


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Excelで文字化けしにくいよう utf-8-sig"""
    if df is None:
        df = pd.DataFrame()
    s = df.to_csv(index=False, encoding="utf-8-sig")
    return s.encode("utf-8-sig")


def _json_to_bytes(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _compute_overall_step2(
    question_entropy_df: pd.DataFrame,
    question_categories_df: pd.DataFrame,
) -> dict:
    """Step2結果から全体統計を作る（卒論・比較実験用）"""
    overall: dict = {"n_samples": int(len(question_entropy_df)) if question_entropy_df is not None else 0}

    if question_entropy_df is not None and not question_entropy_df.empty:
        for col in ["normalized_entropy_questions", "max_category_ratio", "extras_questions"]:
            if col in question_entropy_df.columns and pd.api.types.is_numeric_dtype(question_entropy_df[col]):
                overall[f"{col}_mean"] = _safe_float(question_entropy_df[col].mean())
                overall[f"{col}_min"] = _safe_float(question_entropy_df[col].min())
                overall[f"{col}_max"] = _safe_float(question_entropy_df[col].max())

        # Extras率（samples単位の平均）
        if "extras_questions" in question_entropy_df.columns and "questions" in question_entropy_df.columns:
            q = question_entropy_df["questions"]
            e = question_entropy_df["extras_questions"]
            if pd.api.types.is_numeric_dtype(q) and pd.api.types.is_numeric_dtype(e):
                ratio = (e / q.replace(0, pd.NA)).astype("float64")
                overall["extras_ratio_mean"] = _safe_float(ratio.mean(skipna=True))
                overall["extras_ratio_min"] = _safe_float(ratio.min(skipna=True))
                overall["extras_ratio_max"] = _safe_float(ratio.max(skipna=True))

    # カテゴリ総量（全サンプル合算）も参考として入れる
    if question_categories_df is not None and not question_categories_df.empty:
        if "category" in question_categories_df.columns and "count" in question_categories_df.columns:
            try:
                total = int(question_categories_df["count"].sum())
                overall["total_questions_labeled"] = total
            except Exception:
                pass

    return overall


def _build_step2_zip_bytes(
    *,
    question_entropy_df: pd.DataFrame,
    question_categories_df: pd.DataFrame,
    overall: dict,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("step2_question_entropy_df.csv", _df_to_csv_bytes(question_entropy_df))
        zf.writestr("step2_question_categories_df.csv", _df_to_csv_bytes(question_categories_df))
        zf.writestr("step2_overall.json", _json_to_bytes(overall))
    return buf.getvalue()


# -------------------- UI --------------------
st.set_page_config(page_title="評価ツールB2 Step2（質問カテゴリ分布）", layout="wide")
st.title("評価実験ツールB2 Step2（質問カテゴリ分布）")

st.caption(
    "run_*.zip（試行ZIP）をアップロードし，Step2（質問カテゴリ分類）のみを実行します．"
    "カテゴリ定義（JSON）と分類プロンプト（TXT）はUIから切り替え可能です．"
)

files = st.file_uploader(
    "run_*.zip を複数選択してアップロード",
    type=["zip"],
    accept_multiple_files=True,
)

with st.expander("設定（Step2 prompt / categories / LLM / outputs）", expanded=True):
    st.markdown("#### Step2（質問カテゴリ分布）")

    # ここはあなたのファイル名に合わせてパスを指定
    prompt_path = st.text_input(
        "prompt path (.txt)",
        "prompts/tool_b2_deepdive.txt",
        help="Step2分類用プロンプト。Generation-aligned / Misaligned / Bridged を切り替える場合は別txtを指定。",
    ).strip()

    categories_path = st.text_input(
        "categories path (.json)",
        "configs/Misaligned.json",
        help="カテゴリ定義JSON。例：configs/Generation-aligned.json, configs/Misaligned.json, configs/Bridged.json",
    ).strip()

    temperature = st.number_input("temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    max_retries = st.number_input("max_retries", min_value=0, max_value=5, value=2, step=1)

    default_out = f"outputs/tool_b2_step2_only/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outputs_dir = st.text_input("outputs_dir（ログ保存先）", default_out).strip()

    st.info(
        "Step2はLLMを呼び出します（時間がかかる場合があります）．"
        "outputs_dir 配下に tool_b2_labels.jsonl / tool_b2_errors.jsonl などが保存されます．"
    )

if not files:
    st.info("ZIPをアップロードしてください．")
    st.stop()

try:
    trials = load_trials_from_uploaded_files(files)
except Exception as e:
    st.error("ZIPの読み込みに失敗しました．")
    st.exception(e)
    st.stop()

st.success(f"読み込み成功：{len(trials)} サンプル（ZIP）")

run_clicked = st.button("Step2 を実行", type="primary")

if run_clicked:
    out = Path(outputs_dir)

    with st.status("Step2 実行中…（質問カテゴリ分類）", expanded=True) as status:
        res2 = run_tool_b2_step2_question_distribution(
            trials,
            prompt_path=Path(prompt_path),
            categories_path=Path(categories_path),
            outputs_dir=out,
            temperature=float(temperature),
            max_retries=int(max_retries),
        )
        status.update(label="完了", state="complete", expanded=False)

    # 結果表示
    st.subheader("question_entropy_df（sample_id単位）")
    st.dataframe(res2.question_entropy_df, use_container_width=True)

    st.subheader("question_categories_df（カテゴリ別count：全サンプル合算）")
    st.dataframe(res2.question_categories_df, use_container_width=True)

    overall = _compute_overall_step2(res2.question_entropy_df, res2.question_categories_df)

    st.subheader("Step2 overall（比較実験用の要約）")
    st.json(overall)

    # ZIPダウンロード（Step2 core only）
    zip_bytes = _build_step2_zip_bytes(
        question_entropy_df=res2.question_entropy_df,
        question_categories_df=res2.question_categories_df,
        overall=overall,
    )

    st.download_button(
        label="step2_core_outputs.zip をダウンロード",
        data=zip_bytes,
        file_name="step2_core_outputs.zip",
        mime="application/zip",
    )
