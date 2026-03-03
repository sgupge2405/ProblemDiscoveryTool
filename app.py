# app.py
import streamlit as st

st.set_page_config(page_title="問題発見ツール", layout="wide")
st.title("問題発見ツール（問題意識抽出 → 根拠付与 / 自動インタビュー）")

st.markdown("""
左のサイドバーから工程を選んでください。

- 1) 問題意識抽出
- 2) 根拠付与（実運用想定）
- 3) 自動インタビュー（AI×AI）
- 4) 評価ツールA（堅牢性）
- 5) 評価ツールB1（IR網羅性）
- 5) 評価ツールB2（質問の妥当性）
""")
