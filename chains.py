from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import os
import json
import uuid
import re

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

# -----------------------------------------
# 環境設定
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = os.getenv("OPENAI_MODEL", os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5"))
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------
# プロンプトファイルのパス
# -----------------------------------------

# 第1モジュール：問題意識抽出
PROBLEM_EXTRACT_PROMPT_PATH = BASE_DIR / "prompts" / "problem_extract.md"

# 第2モジュール：根拠付与（インタビュー深掘り）
EVIDENCE_PROMPT_PATH = BASE_DIR / "prompts" / "evidence_interview.md"

# インタビュー受け手のAIエージェント
PERSONA_PROMPT_PATH = BASE_DIR / "prompts" / "persona_answer.md"


# -----------------------------------------
# プロンプト読み込み
# -----------------------------------------
def _load_problem_extract_prompt() -> str:
    """問題意識抽出モジュール用プロンプトを読み込む。"""
    with PROBLEM_EXTRACT_PROMPT_PATH.open("r", encoding="utf-8") as f:
        return f.read()


def _load_evidence_prompt() -> str:
    """根拠付与モジュール用プロンプトを読み込む。"""
    with EVIDENCE_PROMPT_PATH.open("r", encoding="utf-8") as f:
        return f.read()


def _load_persona_prompt() -> str:
    """受け手エージェント（ペルソナ回答）用プロンプトを読み込む。"""
    with PERSONA_PROMPT_PATH.open("r", encoding="utf-8") as f:
        return f.read()


# -----------------------------------------
# 共通 LLM 呼び出しユーティリティ
# -----------------------------------------
def _supports_custom_temperature(model_name: str) -> bool:
    """一部モデルでは temperature がサポートされないので、その判定."""
    if not model_name:
        return False
    blocked_keywords = ["-mini", "realtime", "audio-"]
    return not any(k in model_name for k in blocked_keywords)


def call_llm(model: str, system_prompt: str, user_text: str, temperature: float = 0.2) -> str:
    """
    共通の LLM 呼び出し関数。
    今後ほかのモジュール（問題意識抽出・根拠付与など）からも再利用する。
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }

    if _supports_custom_temperature(model):
        kwargs["temperature"] = temperature

    try:
        resp = _client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        # 一部モデルは temperature 非対応なので、その場合は temperature を外して再実行
        msg = str(e)
        if "temperature" in msg and "unsupported" in msg:
            kwargs.pop("temperature", None)
            resp = _client.chat.completions.create(**kwargs)
        else:
            raise

    return resp.choices[0].message.content


# -----------------------------------------
# JSON抽出（混入テキスト対策）
# -----------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _extract_json_object(text: str) -> str | None:
    """
    LLM出力からJSONオブジェクト部分だけを抽出する。
    - ```json ... ``` があれば最優先
    - なければ最初の { から最後の } までを候補にする
    """
    if not isinstance(text, str) or not text.strip():
        return None

    m = _JSON_FENCE_RE.search(text)
    if m:
        cand = m.group(1).strip()
        if cand.startswith("{") and cand.endswith("}"):
            return cand

    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s

    # 前置き文が混ざるケース：最初の{〜最後の}を拾う
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        cand = s[start : end + 1].strip()
        if cand.startswith("{") and cand.endswith("}"):
            return cand

    return None


# -----------------------------------------
# 第1/第2モジュール共通：JSON(IR) / ASK判定
# -----------------------------------------
def _detect_mode_and_pack(text: str) -> Dict[str, Any]:
    """
    LLM 出力が JSON(IR) か自然文かを判定し、ASK_OR_SUMMARY / IR を返す。

    戻り値:
        mode == "ASK_OR_SUMMARY":
            {"mode": "ASK_OR_SUMMARY", "text": <自然文>} を返す
        mode == "IR":
            {"mode": "IR", "ir": <IR dict>, "raw": <抽出したJSON文字列>} を返す
    """
    json_str = _extract_json_object(text)
    if json_str is None:
        return {"mode": "ASK_OR_SUMMARY", "text": text}

    try:
        ir = json.loads(json_str)
    except Exception:
        # JSON パースに失敗したら自然文扱い
        return {"mode": "ASK_OR_SUMMARY", "text": text}

    # session_id が無ければここで補完
    if isinstance(ir, dict) and not ir.get("session_id"):
        ir["session_id"] = str(uuid.uuid4())

    return {"mode": "IR", "ir": ir, "raw": json_str}


# -----------------------------------------
# 第1モジュール：問題意識抽出用チェーン
# -----------------------------------------
def run_problem_extract(user_text: str) -> Dict[str, Any]:
    """
    第1モジュール：問題意識抽出モジュール用チェーン。

    ユーザー入力を受け取り、
    - mode == "ASK_OR_SUMMARY": 追い質問 or 要約テキスト
    - mode == "IR": objective_card を含む IR(JSON)
    のどちらかを返す。
    """
    prompt = _load_problem_extract_prompt()
    text = call_llm(MODEL_NAME, prompt, user_text)
    return _detect_mode_and_pack(text)


# -----------------------------------------
# 第2モジュール：根拠付与（インタビュー深掘り）用チェーン
# -----------------------------------------
def run_evidence_attach(user_text: str) -> Dict[str, Any]:
    """
    第2モジュール：根拠付与（インタビュー深掘り）用チェーン。
    次に投げる「質問文」または @@ を返す。
    - @@ が返った場合：app側で IR確定(run_evidence_finalize) → 次工程へ進む
    """
    prompt = _load_evidence_prompt()
    text = call_llm(MODEL_NAME, prompt, user_text)
    # app側の extract_question は "text" を見れるが、統一のため question も入れる
    return {"mode": "QUESTION", "text": text, "question": text}


def run_evidence_finalize(user_text: str) -> Dict[str, Any]:
    """
    第2モジュール：根拠付与の“確定”用。
    explanation を追記した IR(JSON) を返す。
    """
    system_prompt = (
        "あなたはソフトウェア企画支援ツールの根拠付与モジュールです。"
        "ユーザー入力には第1モジュールIRとインタビューログが含まれます。"
        "objective_card の各項目に対応する *_explanation に、"
        "ログから要約した根拠を配列要素として追記し、更新後のIR(JSON)を1つだけ返してください。"
        "必ずJSONのみを出力し、説明文は禁止。"
    )
    text = call_llm(MODEL_NAME, system_prompt, user_text)
    return _detect_mode_and_pack(text)


# -----------------------------------------
# インタビュー受け手エージェント用チェーン
# -----------------------------------------
def run_persona_answer(user_text: str) -> Dict[str, Any]:
    """
    受け手エージェント（ペルソナ）用チェーン。
    user_text には「IRの文脈 + 質問」を含めて渡す想定。
    """
    prompt = _load_persona_prompt()
    text = call_llm(MODEL_NAME, prompt, user_text)
    return {"mode": "ANSWER", "text": text}
