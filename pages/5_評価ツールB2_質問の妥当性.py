# pages/6_評価ツールB2_質問妥当性_統合Step1-4.py
"""評価実験ツールB2（質問妥当性）Step1〜4 統合ページ（UI実行）

- run_*.zip を一度だけアップロードして Step1〜4 を連続実行する。
- 中間ZIP（Step1〜4の出力ZIP）の保存/読み込みは行わない。
- 最終的に Step5 と同等の summary_df / overall を画面表示する。
- ダウンロードは「tool_b2_core_outputs.zip」のみ（summary/overallの単体DLはしない）

注意:
- Step2/3/4 は LLM を呼び出す（時間がかかる）。
- outputs base dir 配下に、各StepのログやCSVを保存する。
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
from evaluation.tool_b2_step1 import run_tool_b2_step1_extract_qa
from evaluation.tool_b2_step2 import run_tool_b2_step2_question_distribution
from evaluation.tool_b2_step3 import run_tool_b2_step3_response_alignment_012
from evaluation.tool_b2_step4 import run_tool_b2_step4_answer_meaning_scores


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str = "sample_id") -> pd.DataFrame:
    if left is None or left.empty:
        return right
    if right is None or right.empty:
        return left
    if on not in left.columns or on not in right.columns:
        return left
    return left.merge(right, on=on, how="outer")


def _compute_nonzero_meaning_types(answer_category_sum_df: pd.DataFrame) -> pd.DataFrame:
    """Step4の answer_category_sum_df から非ゼロ意味カテゴリ数を算出する（Step5同等）"""
    if answer_category_sum_df is None or answer_category_sum_df.empty:
        return pd.DataFrame(columns=["sample_id", "answer_meaning_type_count"])

    exclude = {
        "sample_id",
        "total_score",
        "answer_count",
        "normalized_entropy_no_activity",
        "normalized_entropy_all",
        "activity_sum",
        "activity_mean",
        "activity_ratio",
        "activity_saturation",
    }

    score_cols = [c for c in answer_category_sum_df.columns if c not in exclude]
    numeric_cols = [c for c in score_cols if pd.api.types.is_numeric_dtype(answer_category_sum_df[c])]

    if not numeric_cols:
        return pd.DataFrame({"sample_id": answer_category_sum_df["sample_id"], "answer_meaning_type_count": 0})

    tmp = answer_category_sum_df[["sample_id"] + numeric_cols].copy()
    tmp["answer_meaning_type_count"] = (tmp[numeric_cols] > 0).sum(axis=1)
    return tmp[["sample_id", "answer_meaning_type_count"]]


def _compute_overall_stats(summary_df: pd.DataFrame) -> dict:
    overall = {"n_samples": int(len(summary_df))}

    def add_stats(col: str) -> None:
        if col in summary_df.columns and pd.api.types.is_numeric_dtype(summary_df[col]):
            overall[f"{col}_mean"] = float(summary_df[col].mean())
            overall[f"{col}_min"] = float(summary_df[col].min())
            overall[f"{col}_max"] = float(summary_df[col].max())

    for col in [
        "question_turns",
        "normalized_entropy_questions",
        "alignment_rate",  # Step3: score>0率（2は1に潰した一致率）
        "unanswered_pairs",
        "answer_total_score",
        "normalized_entropy_no_activity",
        "activity_ratio",
        "answer_meaning_type_count",
        "max_category_ratio",
        # 追加で見たい場合（新Step3で出る列）
        "score_mean",
        "score2_rate",
        "failed_pairs",
    ]:
        add_stats(col)

    return overall


def _merge_step_outputs(
    *,
    turn_stats_df: pd.DataFrame,
    question_entropy_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    answer_entropy_df: pd.DataFrame,
    answer_category_sum_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Step5相当の結合（ZIP無し／メモリ上のDFを統合）"""

    summary_df = pd.DataFrame()

    # Step1
    if not turn_stats_df.empty and "sample_id" in turn_stats_df.columns:
        keep = [
            "sample_id",
            "question_turns",
            "answer_turns",
            "complete_pairs",
            "total_turns",
            "missing_question_turns",
            "missing_answer_turns",
        ]
        keep = [c for c in keep if c in turn_stats_df.columns]
        summary_df = _safe_merge(summary_df, turn_stats_df[keep])

    # Step2
    if not question_entropy_df.empty and "sample_id" in question_entropy_df.columns:
        summary_df = _safe_merge(summary_df, question_entropy_df)

    # Step3（応答整合）
    if not alignment_df.empty and "sample_id" in alignment_df.columns:
        summary_df = _safe_merge(summary_df, alignment_df)

    # Step4（entropy）
    if not answer_entropy_df.empty and "sample_id" in answer_entropy_df.columns:
        df = answer_entropy_df.copy()
        if "total_score" in df.columns:
            df = df.rename(columns={"total_score": "answer_total_score"})
        summary_df = _safe_merge(summary_df, df)

    # Step4（意味カテゴリ種類数）
    if not answer_category_sum_df.empty and "sample_id" in answer_category_sum_df.columns:
        types_df = _compute_nonzero_meaning_types(answer_category_sum_df)
        summary_df = _safe_merge(summary_df, types_df)

    if "sample_id" in summary_df.columns:
        summary_df = summary_df.sort_values("sample_id").reset_index(drop=True)

    overall = _compute_overall_stats(summary_df)
    return summary_df, overall


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Excelで文字化けしにくいよう utf-8-sig を使う"""
    if df is None:
        df = pd.DataFrame()
    s = df.to_csv(index=False, encoding="utf-8-sig")
    return s.encode("utf-8-sig")


def _json_to_bytes(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def _build_core_zip_bytes(
    *,
    summary_df: pd.DataFrame,
    overall: dict,
    # Step1
    step1_turn_stats_df: pd.DataFrame,
    step1_summary: dict,
    # Step2
    step2_question_entropy_df: pd.DataFrame,
    step2_question_categories_df: pd.DataFrame,
    # Step3
    step3_alignment_df: pd.DataFrame,
    step3_alignment_detail_df: pd.DataFrame,
    # Step4
    step4_answer_entropy_df: pd.DataFrame,
    step4_answer_category_sum_df: pd.DataFrame,
) -> bytes:
    """あなたの指定した core 一式だけを詰めたZIPを生成"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 統合
        zf.writestr("tool_b2_summary_df.csv", _df_to_csv_bytes(summary_df))
        zf.writestr("tool_b2_overall.json", _json_to_bytes(overall))

        # Step1
        zf.writestr("step1_turn_stats_df.csv", _df_to_csv_bytes(step1_turn_stats_df))
        zf.writestr("step1_summary.json", _json_to_bytes(step1_summary))

        # Step2
        zf.writestr("step2_question_entropy_df.csv", _df_to_csv_bytes(step2_question_entropy_df))
        zf.writestr("step2_question_categories_df.csv", _df_to_csv_bytes(step2_question_categories_df))

        # Step3
        zf.writestr("step3_alignment_df.csv", _df_to_csv_bytes(step3_alignment_df))
        zf.writestr("step3_alignment_detail_df.csv", _df_to_csv_bytes(step3_alignment_detail_df))

        # Step4
        zf.writestr("step4_answer_entropy_df.csv", _df_to_csv_bytes(step4_answer_entropy_df))
        zf.writestr("step4_answer_category_sum_df.csv", _df_to_csv_bytes(step4_answer_category_sum_df))

    return buf.getvalue()


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="評価ツールB2（統合Step1-4）", layout="wide")
st.title("評価実験ツールB2（質問妥当性）Step1〜4 統合実行")

st.caption(
    "run_*.zip（試行ZIP）を1回だけアップロードし，Step1〜4を連続実行して，"
    "Step5相当の統合表（summary_df）を表示します．（中間ZIPの保存/読み込みはしません）"
)

files = st.file_uploader(
    "run_*.zip を複数選択してアップロード",
    type=["zip"],
    accept_multiple_files=True,
)

with st.expander("設定（LLMプロンプト／カテゴリ／出力ログ）", expanded=True):
    # Step2
    st.markdown("#### Step2（質問カテゴリ分布）")
    prompt_step2 = st.text_input("Step2 prompt path", "prompts/tool_b2_deepdive.txt").strip()
    categories_step2 = st.text_input(
        "Step2 categories path", "configs/tool_b2_deepdive_categories.json"
    ).strip()

    # Step3
    st.markdown("#### Step3（応答整合：0/1/2 + 理由）")
    judge_prompt_step3 = st.text_input(
        "Step3 judge prompt path",
        "prompts/tool_b2_response_alignment_012.txt",
        help="質問と回答を入力として，score(0/1/2)とreasonをJSONで返す評価プロンプト",
    ).strip()

    # Step4
    st.markdown("#### Step4（回答意味カテゴリ）")
    prompt_step4 = st.text_input("Step4 prompt path", "prompts/tool_b_meaning.txt").strip()
    categories_step4 = st.text_input("Step4 categories path", "configs/tool_b_categories.json").strip()

    st.markdown("#### 共通設定")
    temperature = st.number_input("temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    max_retries = st.number_input("max_retries", min_value=0, max_value=5, value=2, step=1)

    default_out_base = f"outputs/tool_b2_question_validity_allinone/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outputs_base = st.text_input("outputs base dir（ログ保存先）", default_out_base).strip()

    st.info(
        "Step2/3/4はLLM呼び出しが走るため時間がかかります．"
        "outputs base dir 配下に，jsonlログ（labels/errors）およびCSV（alignment等）を保存します．"
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

run_clicked = st.button("Step1〜4 を連続実行して統合（Step5相当）", type="primary")

if run_clicked:
    base = Path(outputs_base)
    out2 = base / "step2"
    out3 = base / "step3"
    out4 = base / "step4"

    with st.status("実行中…（Step1→2→3→4）", expanded=True) as status:
        # Step1
        status.write("Step1：QA抽出…")
        res1 = run_tool_b2_step1_extract_qa(trials)

        # Step2
        status.write("Step2：質問カテゴリ分布（LLM）…")
        res2 = run_tool_b2_step2_question_distribution(
            trials,
            prompt_path=Path(prompt_step2),
            categories_path=Path(categories_step2),
            outputs_dir=out2,
            temperature=float(temperature),
            max_retries=int(max_retries),
        )

        # Step3
        status.write("Step3：応答整合（0/1/2 + 理由）（LLM）…")
        res3 = run_tool_b2_step3_response_alignment_012(
            trials,
            judge_prompt_path=Path(judge_prompt_step3),
            outputs_dir=out3,
            temperature=float(temperature),
            max_retries=int(max_retries),
        )

        # Step4
        status.write("Step4：回答意味カテゴリスコア（LLM）…")
        res4 = run_tool_b2_step4_answer_meaning_scores(
            trials,
            prompt_path=Path(prompt_step4),
            categories_path=Path(categories_step4),
            outputs_dir=out4,
            temperature=float(temperature),
            max_retries=int(max_retries),
        )

        # Merge (Step5相当)
        status.write("統合（Step5相当）：sample_id 結合…")
        summary_df, overall = _merge_step_outputs(
            turn_stats_df=res1.turn_stats_df,
            question_entropy_df=res2.question_entropy_df,
            alignment_df=res3.alignment_df,
            answer_entropy_df=res4.answer_entropy_df,
            answer_category_sum_df=res4.answer_category_sum_df,
        )

        status.update(label="完了", state="complete", expanded=False)

    st.subheader("統合結果（sample_id単位のまとめ表：Step5相当）")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("全体統計（平均・最小・最大）")
    st.json(overall)

    with st.expander("（参考）Step別の中間出力", expanded=False):
        st.markdown("#### Step1：ターン統計")
        st.dataframe(res1.turn_stats_df, use_container_width=True)

        st.markdown("#### Step2：質問Entropy")
        st.dataframe(res2.question_entropy_df, use_container_width=True)

        st.markdown("#### Step3：応答整合")
        st.dataframe(res3.alignment_df, use_container_width=True)

        st.markdown("#### Step4：回答Entropy")
        st.dataframe(res4.answer_entropy_df, use_container_width=True)

    # --------------------
    # ZIP（core outputs only / 指定一式）
    # --------------------
    # Step1 summary（mean/min/max等）
    # Step1の戻り値が summary(dict) を持つ前提。もし無い場合は例外として落としてよい（仕様の一貫性重視）
    step1_summary = res1.summary  # dict

    # Step2 categories df
    step2_question_categories_df = res2.question_categories_df

    # Step3 detail df（理由）
    step3_alignment_detail_df = res3.alignment_detail_df

    zip_bytes = _build_core_zip_bytes(
        summary_df=summary_df,
        overall=overall,
        # Step1
        step1_turn_stats_df=res1.turn_stats_df,
        step1_summary=step1_summary,
        # Step2
        step2_question_entropy_df=res2.question_entropy_df,
        step2_question_categories_df=step2_question_categories_df,
        # Step3
        step3_alignment_df=res3.alignment_df,
        step3_alignment_detail_df=step3_alignment_detail_df,
        # Step4
        step4_answer_entropy_df=res4.answer_entropy_df,
        step4_answer_category_sum_df=res4.answer_category_sum_df,
    )

    st.download_button(
        label="tool_b2_core_outputs.zip をダウンロード",
        data=zip_bytes,
        file_name="tool_b2_core_outputs.zip",
        mime="application/zip",
    )
