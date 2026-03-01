# evaluation/tool_a.py
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    # OpenAI公式SDK（OpenAI() / client.embeddings.create）
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    # 長文を安全にチャンク化するため（入ってなくても動く）
    import tiktoken
except Exception:
    tiktoken = None  # type: ignore


# -----------------------------
# Config
# -----------------------------

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CACHE_PATH = ".cache/embeddings.sqlite"

# 評価対象から除外するIR項目（インタビュー由来ではない/付加情報）
EXCLUDE_FIELDS = {"refs", "quality"}

# explanation重視（あなたの設計に合わせる）
# - *_explanation のみを評価対象にする（True推奨）
DEFAULT_EXPLANATION_ONLY = True

# 「未反映候補」抽出の閾値（explanation との最大類似度がこれ未満なら未反映候補）
DEFAULT_UNCOVERED_THRESHOLD = 0.40

# 未反映候補を作るときの、1 trial あたり最大出力数（無制限にすると重くなるため安全弁）
DEFAULT_UNCOVERED_MAX_ITEMS_PER_TRIAL = 300


# -----------------------------
# Helpers: text normalization
# -----------------------------

def _to_text(v: Any) -> str:
    """IR各項目の値を評価用テキストに整形（空なら空文字）"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v)
    # list[str] などは JSON で潰すより、後で join した方が扱いやすいのでここでは文字列化しない
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(v)


def _safe_join_text(parts: Iterable[str]) -> str:
    xs = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return "\n".join(xs)


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def _pairwise_cosine_stats(vectors: List[np.ndarray]) -> Tuple[float, float, int]:
    """
    全組合せ(i<j)のコサイン類似度の平均・分散・ペア数を返す
    """
    n = len(vectors)
    if n < 2:
        return float("nan"), float("nan"), 0

    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine(vectors[i], vectors[j]))

    arr = np.array(sims, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.var(ddof=0)), int(arr.size)


# -----------------------------
# Token helpers (chunking for long texts)
# -----------------------------

def _get_token_encoder(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _estimate_tokens(text: str, model: str) -> int:
    enc = _get_token_encoder(model)
    if enc is None:
        # フォールバック：安全側に見積もる（日本語はトークンが増えやすい）
        return max(1, int(len(text) * 0.9))
    return len(enc.encode(text))


def _chunk_text_by_tokens(text: str, model: str, max_tokens: int) -> List[str]:
    """
    可能なら tiktoken でトークン単位に分割。
    tiktoken が無ければ文字数で粗く分割（安全側）。
    """
    text = (text or "").strip()
    if not text:
        return []

    enc = _get_token_encoder(model)
    if enc is None:
        # 文字数で分割（max_tokens を文字上限目安として流用）
        step = max(500, int(max_tokens * 0.9))
        return [text[i : i + step] for i in range(0, len(text), step)]

    toks = enc.encode(text)
    chunks: List[str] = []
    for i in range(0, len(toks), max_tokens):
        chunks.append(enc.decode(toks[i : i + max_tokens]))
    return chunks


def _average_embeddings(vecs: List[np.ndarray]) -> np.ndarray:
    """
    チャンク埋め込みを平均して1本にする。
    """
    xs = [v for v in vecs if isinstance(v, np.ndarray) and v.size > 0]
    if not xs:
        return np.array([], dtype=np.float32)
    arr = np.vstack(xs)
    return arr.mean(axis=0).astype(np.float32)


# -----------------------------
# Embedding cache (SQLite)
# -----------------------------

class EmbeddingCache:
    def __init__(self, path: str = DEFAULT_CACHE_PATH) -> None:
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def _hash_key(model: str, text: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\n")
        h.update(text.encode("utf-8", errors="replace"))
        return h.hexdigest()

    def get(self, model: str, text: str) -> Optional[np.ndarray]:
        key = self._hash_key(model, text)
        cur = self.conn.cursor()
        cur.execute("SELECT dim, vec FROM embeddings WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        dim, blob = row
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != dim:
            return None
        return arr.astype(np.float32)

    def set(self, model: str, text: str, vec: np.ndarray) -> None:
        key = self._hash_key(model, text)
        vec32 = np.asarray(vec, dtype=np.float32)
        blob = vec32.tobytes()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO embeddings(key, model, dim, vec, created_at) VALUES(?,?,?,?,?)",
            (key, model, int(vec32.size), blob, int(time.time())),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# -----------------------------
# Condition handling
# -----------------------------

def build_condition_label(meta: Dict[str, Any], keys: Sequence[str]) -> str:
    """
    条件ラベルを meta.json の指定キーから構成する。
    例：keys=["admin_rag"] → "admin_rag=True"
    """
    if not keys:
        return "all"
    parts = []
    for k in keys:
        if k in meta:
            parts.append(f"{k}={meta.get(k)}")
        else:
            parts.append(f"{k}=<missing>")
    return "|".join(parts)


# -----------------------------
# Field extraction from IR
# -----------------------------

def _is_explanation_field(key: str) -> bool:
    return isinstance(key, str) and key.endswith("_explanation")


def extract_ir_fields(trial, field_keys: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """
    loader.py の TrialRecord を受け取り、objective_cardの各項目をテキスト化して返す。
    - *_explanation は「配列 or 文字列」を想定し、要素を改行連結してテキスト化する。
    """
    oc = getattr(trial, "objective_card", {})
    if not isinstance(oc, dict):
        return {}

    keys = list(oc.keys()) if field_keys is None else list(field_keys)

    out: Dict[str, str] = {}
    for k in keys:
        if k not in oc:
            continue
        v = oc.get(k)

        # explanation は list[str] を想定（あなたのIR実例）
        if _is_explanation_field(k):
            if isinstance(v, list):
                out[k] = _safe_join_text([str(x) for x in v])
            elif isinstance(v, str):
                out[k] = v.strip()
            else:
                out[k] = _to_text(v)
        else:
            # 非explanation は文字列化のみ（今回の主戦場ではない）
            out[k] = _to_text(v)

    return out


def decide_field_keys(trials: List[Any], prefer: str = "intersection") -> List[str]:
    """
    objective_card の項目集合を決める。
    prefer:
      - "intersection": 全trialに共通するキーのみ
      - "union": 出現したキーをすべて
    """
    key_sets: List[set] = []
    for t in trials:
        oc = getattr(t, "objective_card", {})
        if isinstance(oc, dict):
            key_sets.append(set(oc.keys()))
    if not key_sets:
        return []
    if prefer == "union":
        keys = set().union(*key_sets)
    else:
        keys = set.intersection(*key_sets)
    return sorted(list(keys))


def build_interviewee_log_text(trial) -> str:
    """
    情報保持用：interviewee（回答）発話を連結して L を作る。
    loader.py が answers を持っている場合はそれを優先。
    """
    if hasattr(trial, "answers") and getattr(trial, "answers"):
        return _safe_join_text(trial.answers)

    parts: List[str] = []
    for r in getattr(trial, "chat_rows", []):
        role = getattr(r, "role", "")
        text = getattr(r, "text", "")
        if role in ("interviewee", "user", "persona", "answerer") and text:
            parts.append(text)
    return _safe_join_text(parts)


def build_explanations_all_text(fields: Dict[str, str], field_keys: Sequence[str]) -> str:
    """
    *_explanation をまとめた全文 E を作る（explanation重視の保持指標用）。
    """
    parts: List[str] = []
    for k in field_keys:
        if _is_explanation_field(k):
            t = (fields.get(k, "") or "").strip()
            if t:
                parts.append(f"[{k}]\n{t}")
    return _safe_join_text(parts)


# -----------------------------
# Embedding (with long-text chunking)
# -----------------------------

def embed_texts(
    texts: List[str],
    model: str,
    cache: EmbeddingCache,
    client: Any,
    batch_size: int = 64,
    max_input_tokens: int = 8192,
    safety_margin: int = 256,
) -> List[np.ndarray]:
    """
    texts を埋め込みし、np.ndarray のリストで返す（キャッシュあり）。

    - 1入力が max_input_tokens を超える場合：
      チャンク化して埋め込み → 平均して1本に戻す（切り捨てない）。
    """
    if OpenAI is None or client is None:
        raise RuntimeError("openai SDK が import できません。`pip install openai` を確認してください。")

    per_text_limit = max(256, max_input_tokens - safety_margin)
    out_vecs: List[np.ndarray] = []

    for tx in texts:
        tx = (tx or "").strip()
        if not tx:
            out_vecs.append(np.array([], dtype=np.float32))
            continue

        tok = _estimate_tokens(tx, model=model)

        # 通常ルート
        if tok <= per_text_limit:
            v = cache.get(model, tx)
            if v is not None:
                out_vecs.append(v)
            else:
                resp = client.embeddings.create(model=model, input=[tx])
                emb = np.array(resp.data[0].embedding, dtype=np.float32)
                cache.set(model, tx, emb)
                out_vecs.append(emb)
            continue

        # 長文ルート：チャンク化して平均
        chunks = _chunk_text_by_tokens(tx, model=model, max_tokens=per_text_limit)

        # キャッシュヒット/ミスを分ける
        chunk_vecs_tmp: List[Optional[np.ndarray]] = [None] * len(chunks)
        pending_texts: List[str] = []
        pending_indices: List[int] = []

        for i, c in enumerate(chunks):
            c = (c or "").strip()
            if not c:
                continue
            cv = cache.get(model, c)
            if cv is not None:
                chunk_vecs_tmp[i] = cv
            else:
                pending_texts.append(c)
                pending_indices.append(i)

        # ミス分だけバッチでAPIへ
        if pending_texts:
            for start in range(0, len(pending_texts), batch_size):
                sub_texts = pending_texts[start : start + batch_size]
                resp = client.embeddings.create(model=model, input=sub_texts)
                for j, item in enumerate(resp.data):
                    emb = np.array(item.embedding, dtype=np.float32)
                    ctext = sub_texts[j]
                    cache.set(model, ctext, emb)
                    idx = pending_indices[start + j]
                    chunk_vecs_tmp[idx] = emb

        vecs = [v for v in chunk_vecs_tmp if v is not None and v.size > 0]
        out_vecs.append(_average_embeddings(vecs))

    return out_vecs


# -----------------------------
# Uncovered candidates (log not reflected in explanations)
# -----------------------------

def _split_units_for_uncovered(text: str) -> List[str]:
    """
    ログから「未反映候補」を取るための最小分割。
    - 改行で分割し、短すぎるものを除く
    """
    lines = [x.strip() for x in (text or "").splitlines()]
    # 短すぎる断片はノイズになりやすいので除外（調整可）
    units = [x for x in lines if len(x) >= 10]
    return units


def extract_uncovered_candidates_for_trial(
    run_id: str,
    condition: str,
    L_text: str,
    explanations_all_text: str,
    model: str,
    cache: EmbeddingCache,
    client: Any,
    threshold: float = DEFAULT_UNCOVERED_THRESHOLD,
    max_items: int = DEFAULT_UNCOVERED_MAX_ITEMS_PER_TRIAL,
) -> List[Dict[str, Any]]:
    """
    ログLの単位（行）ごとに、explanations_all_text(E)と「十分近いか」を判定し、
    近くないものを未反映候補として返す（jsonl出力用）。

    厳密な差分ではなく「意味的な未反映候補」である点が重要。
    """
    L_units = _split_units_for_uncovered(L_text)
    if not L_units:
        return []

    # Eをチャンク化し、unit→E_chunk の最大類似度で判定（E全体1本より精度が出る）
    # embed_textsは内部で長文チャンク化をするが、ここでは「E自体を複数chunkにしてmax」を取る
    # ただし tiktoken が無い場合もあるので token推定で安全側に分割
    E_chunks = _chunk_text_by_tokens(explanations_all_text, model=model, max_tokens=800)
    E_chunks = [c.strip() for c in E_chunks if c.strip()]
    if not E_chunks:
        # explanationが空なら、ログは全部「未反映候補」になり得るが、出力爆発を避けて上限
        out = []
        for u in L_units[:max_items]:
            out.append({
                "run_id": run_id,
                "condition": condition,
                "unit": u,
                "max_sim_to_explanations": float("nan"),
                "threshold": threshold,
                "label": "uncovered_candidate",
                "note": "explanations_all_text_empty",
            })
        return out

    # 埋め込み
    unit_vecs = embed_texts(L_units, model=model, cache=cache, client=client)
    chunk_vecs = embed_texts(E_chunks, model=model, cache=cache, client=client)

    out: List[Dict[str, Any]] = []
    for u, uv in zip(L_units, unit_vecs):
        if uv.size == 0:
            continue
        best = float("nan")
        for cv in chunk_vecs:
            if cv.size == 0:
                continue
            s = _cosine(uv, cv)
            if np.isnan(s):
                continue
            if np.isnan(best) or s > best:
                best = s

        if np.isnan(best) or best < threshold:
            out.append({
                "run_id": run_id,
                "condition": condition,
                "unit": u,
                "max_sim_to_explanations": best,
                "threshold": threshold,
                "label": "uncovered_candidate",
            })
            if len(out) >= max_items:
                break

    return out


# -----------------------------
# Tool A core
# -----------------------------

@dataclass
class ToolAResult:
    summary_df: pd.DataFrame
    field_scores_df: pd.DataFrame
    trial_scores_df: pd.DataFrame
    uncovered_candidates: List[Dict[str, Any]]
    report: Dict[str, Any]


def run_tool_a(
    trials: List[Any],
    condition_keys: Optional[Sequence[str]] = None,
    field_key_policy: str = "intersection",  # intersection / union
    embed_model: str = DEFAULT_EMBED_MODEL,
    cache_path: str = DEFAULT_CACHE_PATH,
    explanation_only: bool = DEFAULT_EXPLANATION_ONLY,
    uncovered_threshold: float = DEFAULT_UNCOVERED_THRESHOLD,
    uncovered_max_items_per_trial: int = DEFAULT_UNCOVERED_MAX_ITEMS_PER_TRIAL,
) -> ToolAResult:
    """
    評価実験ツールA（explanation重視版）：
      - 堅牢性：
          (a) explanation各項目の trial間 類似度（平均/分散）
          (b) explanation全体（*_explanationを連結）同士の trial間 類似度（平均/分散）を summary に追加
      - 情報保持：
          (a) L（intervieweeログ）と explanation各項目の類似度（trial×field）
          (b) L と explanation全体（連結）の類似度 retention_expl_all（trial単位）を trial_scores に追加
      - 未反映候補：
          ログLの単位（行）ごとに explanation と十分近いかを判定し、近くないものを jsonl 用に抽出

    ※ refs / quality は除外する。
    ※ explanation_only=True の場合、*_explanation だけを評価対象とする。
    """
    if not trials:
        raise ValueError("trials が空です。")

    if condition_keys is None:
        has_admin_rag = any(isinstance(getattr(t, "meta", {}), dict) and ("admin_rag" in t.meta) for t in trials)
        condition_keys = ["admin_rag"] if has_admin_rag else []

    # 対象フィールド（objective_card）
    field_keys = decide_field_keys(trials, prefer=field_key_policy)
    if not field_keys:
        raise ValueError("objective_card の評価対象フィールドが決まりませんでした。")

    # refs / quality など、評価対象外フィールドを除外
    field_keys = [k for k in field_keys if k not in EXCLUDE_FIELDS]
    if not field_keys:
        raise ValueError("除外後、評価対象フィールドが空になりました。")

    # explanation重視：*_explanation だけを残す
    if explanation_only:
        field_keys = [k for k in field_keys if _is_explanation_field(k)]
        if not field_keys:
            raise ValueError("explanation_only=True ですが、*_explanation が見つかりませんでした。")

    # trialごとの整形
    trial_rows = []
    for t in trials:
        meta = getattr(t, "meta", {}) or {}
        cond = build_condition_label(meta, condition_keys)
        L = build_interviewee_log_text(t)
        fields = extract_ir_fields(t, field_keys=field_keys)
        E_all = build_explanations_all_text(fields, field_keys)

        trial_rows.append({
            "zip": getattr(t, "zip_name", ""),
            "run_id": getattr(t, "run_id", ""),
            "condition": cond,
            "meta": meta,
            "L_text": L,
            "fields": fields,          # dict[field]->text
            "E_all_text": E_all,       # explanations combined text
        })

    client = OpenAI() if OpenAI is not None else None
    cache = EmbeddingCache(cache_path)

    report: Dict[str, Any] = {
        "tool": "evaluation_tool_A",
        "created_utc": int(time.time()),
        "embed_model": embed_model,
        "condition_keys": list(condition_keys),
        "field_key_policy": field_key_policy,
        "field_keys": field_keys,
        "explanation_only": explanation_only,
        "uncovered_threshold": uncovered_threshold,
        "uncovered_max_items_per_trial": uncovered_max_items_per_trial,
        "n_trials_input": len(trials),
        "excluded_trials": [],
    }

    # -------------------------
    # 1) Retention (trial-wise)
    # -------------------------
    L_texts = [r["L_text"] for r in trial_rows]
    L_vecs = embed_texts(L_texts, model=embed_model, cache=cache, client=client)

    # explanation全体 E_all の保持（trial単位）
    E_all_texts = [r["E_all_text"] for r in trial_rows]
    E_all_vecs = embed_texts(E_all_texts, model=embed_model, cache=cache, client=client)

    retention_expl_all_by_trial: List[float] = []
    for lv, ev in zip(L_vecs, E_all_vecs):
        if lv.size == 0 or ev.size == 0:
            retention_expl_all_by_trial.append(float("nan"))
        else:
            retention_expl_all_by_trial.append(_cosine(lv, ev))

    # 項目別（trial×field）
    field_texts_flat: List[str] = []
    index_map: List[Tuple[int, str]] = []  # (trial_idx, field)
    for i, r in enumerate(trial_rows):
        for f in field_keys:
            field_texts_flat.append((r["fields"].get(f, "") or "").strip())
            index_map.append((i, f))

    field_vecs_flat = embed_texts(field_texts_flat, model=embed_model, cache=cache, client=client)

    retention_by_trial_field: Dict[Tuple[int, str], float] = {}
    for k, (ti, f) in enumerate(index_map):
        lv = L_vecs[ti]
        fv = field_vecs_flat[k]
        if lv.size == 0 or fv.size == 0:
            retention_by_trial_field[(ti, f)] = float("nan")
        else:
            retention_by_trial_field[(ti, f)] = _cosine(lv, fv)

    trial_score_rows = []
    for i, r in enumerate(trial_rows):
        sims = []
        per_field = {}
        for f in field_keys:
            s = retention_by_trial_field.get((i, f), float("nan"))
            per_field[f"retention_{f}"] = s
            if not np.isnan(s):
                sims.append(s)
        # 従来：項目平均
        retention_mean_fields = float(np.mean(sims)) if sims else float("nan")
        # 追加：explanation全体
        retention_expl_all = retention_expl_all_by_trial[i]

        trial_score_rows.append({
            "run_id": r["run_id"],
            "zip": r["zip"],
            "condition": r["condition"],
            "retention_mean_fields": retention_mean_fields,
            "retention_expl_all": retention_expl_all,
            "n_fields_valid": int(len(sims)),
            "L_len": len(r["L_text"]),
            "E_all_len": len(r["E_all_text"]),
            **per_field,
        })

    trial_scores_df = pd.DataFrame(trial_score_rows)

    # -------------------------
    # 2) Robustness (pairwise within condition)
    # -------------------------
    cond_to_indices: Dict[str, List[int]] = {}
    for i, r in enumerate(trial_rows):
        cond_to_indices.setdefault(r["condition"], []).append(i)

    # field: trial×field のベクトル辞書
    field_vec_by_trial_field: Dict[Tuple[int, str], np.ndarray] = {}
    for k, (ti, f) in enumerate(index_map):
        field_vec_by_trial_field[(ti, f)] = field_vecs_flat[k]

    field_score_rows = []
    for cond, idxs in cond_to_indices.items():
        # explanation全体の堅牢性（E_all同士）
        e_vecs = []
        for ti in idxs:
            v = E_all_vecs[ti]
            if isinstance(v, np.ndarray) and v.size > 0:
                e_vecs.append(v)
        e_mean, e_var, e_pairs = _pairwise_cosine_stats(e_vecs)

        # 各 explanation 項目
        for f in field_keys:
            vecs = []
            for ti in idxs:
                v = field_vec_by_trial_field.get((ti, f), np.array([], dtype=np.float32))
                if v.size > 0:
                    vecs.append(v)

            mean_sim, var_sim, n_pairs = _pairwise_cosine_stats(vecs)

            # retention（このcond内の当該fieldの trial別 sim）
            ret_sims = []
            for ti in idxs:
                s = retention_by_trial_field.get((ti, f), float("nan"))
                if not np.isnan(s):
                    ret_sims.append(s)
            ret_mean = float(np.mean(ret_sims)) if ret_sims else float("nan")
            ret_var = float(np.var(ret_sims, ddof=0)) if ret_sims else float("nan")

            field_score_rows.append({
                "condition": cond,
                "field": f,
                "robustness_mean": mean_sim,
                "robustness_var": var_sim,
                "robustness_pairs": n_pairs,
                "n_trials_valid_for_robustness": len(vecs),
                "retention_mean": ret_mean,
                "retention_var": ret_var,
                "n_trials_valid_for_retention": len(ret_sims),
                # 参考：explanation全体の堅牢性（同じcondなら全field行に同値を持たせる）
                "robustness_expl_all_mean": e_mean,
                "robustness_expl_all_var": e_var,
                "robustness_expl_all_pairs": e_pairs,
            })

    field_scores_df = pd.DataFrame(field_score_rows)

    # -------------------------
    # 3) Summary (per condition)
    # -------------------------
    summary_rows = []
    for cond, idxs in cond_to_indices.items():
        sub = field_scores_df[field_scores_df["condition"] == cond]

        rob_means = sub["robustness_mean"].to_numpy(dtype=float)
        rob_vars = sub["robustness_var"].to_numpy(dtype=float)
        rob_means = rob_means[~np.isnan(rob_means)]
        rob_vars = rob_vars[~np.isnan(rob_vars)]

        robustness_mean_over_fields = float(np.mean(rob_means)) if rob_means.size else float("nan")
        robustness_var_over_fields = float(np.mean(rob_vars)) if rob_vars.size else float("nan")

        # explanation全体の堅牢性（cond内で同値なので先頭を採用）
        if len(sub) > 0:
            expl_all_rob_mean = float(sub["robustness_expl_all_mean"].iloc[0])
            expl_all_rob_var = float(sub["robustness_expl_all_var"].iloc[0])
            expl_all_rob_pairs = int(sub["robustness_expl_all_pairs"].iloc[0])
        else:
            expl_all_rob_mean = float("nan")
            expl_all_rob_var = float("nan")
            expl_all_rob_pairs = 0

        # retention：項目平均（trial単位）と、explanation全体（trial単位）
        ts_fields = trial_scores_df[trial_scores_df["condition"] == cond]["retention_mean_fields"].to_numpy(dtype=float)
        ts_fields = ts_fields[~np.isnan(ts_fields)]
        retention_fields_mean = float(np.mean(ts_fields)) if ts_fields.size else float("nan")
        retention_fields_var = float(np.var(ts_fields, ddof=0)) if ts_fields.size else float("nan")

        ts_all = trial_scores_df[trial_scores_df["condition"] == cond]["retention_expl_all"].to_numpy(dtype=float)
        ts_all = ts_all[~np.isnan(ts_all)]
        retention_expl_all_mean = float(np.mean(ts_all)) if ts_all.size else float("nan")
        retention_expl_all_var = float(np.var(ts_all, ddof=0)) if ts_all.size else float("nan")

        summary_rows.append({
            "condition": cond,
            "n_trials": len(idxs),
            "n_fields": len(field_keys),

            "robustness_mean_over_fields": robustness_mean_over_fields,
            "robustness_var_over_fields": robustness_var_over_fields,

            "robustness_expl_all_mean": expl_all_rob_mean,
            "robustness_expl_all_var": expl_all_rob_var,
            "robustness_expl_all_pairs": expl_all_rob_pairs,

            "retention_fields_mean_over_trials": retention_fields_mean,
            "retention_fields_var_over_trials": retention_fields_var,

            "retention_expl_all_mean_over_trials": retention_expl_all_mean,
            "retention_expl_all_var_over_trials": retention_expl_all_var,
        })

    summary_df = pd.DataFrame(summary_rows)

    # -------------------------
    # 4) Uncovered candidates (jsonl)
    # -------------------------
    uncovered_candidates: List[Dict[str, Any]] = []
    for i, r in enumerate(trial_rows):
        cand = extract_uncovered_candidates_for_trial(
            run_id=str(r["run_id"]),
            condition=str(r["condition"]),
            L_text=str(r["L_text"]),
            explanations_all_text=str(r["E_all_text"]),
            model=embed_model,
            cache=cache,
            client=client,
            threshold=uncovered_threshold,
            max_items=uncovered_max_items_per_trial,
        )
        uncovered_candidates.extend(cand)

    report["conditions"] = [{"condition": c, "n_trials": len(idxs)} for c, idxs in cond_to_indices.items()]
    report["trials"] = [
        {
            "run_id": r["run_id"],
            "zip": r["zip"],
            "condition": r["condition"],
            "meta_keys": list(r["meta"].keys()) if isinstance(r["meta"], dict) else [],
            "L_len": len(r["L_text"]),
            "E_all_len": len(r["E_all_text"]),
            "retention_expl_all": retention_expl_all_by_trial[i] if i < len(retention_expl_all_by_trial) else float("nan"),
        }
        for i, r in enumerate(trial_rows)
    ]
    report["uncovered_candidates_total"] = len(uncovered_candidates)

    cache.close()
    return ToolAResult(
        summary_df=summary_df,
        field_scores_df=field_scores_df,
        trial_scores_df=trial_scores_df,
        uncovered_candidates=uncovered_candidates,
        report=report,
    )


# -----------------------------
# Utilities: save outputs
# -----------------------------

def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_tool_a_outputs(
    result: ToolAResult,
    out_dir: str,
    prefix: str = "tool_a",
) -> Dict[str, str]:
    """
    summary.csv / field_scores.csv / trial_scores.csv / report.json / uncovered_candidates.jsonl を保存する。
    戻り値はファイルパス辞書。
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    p1 = os.path.join(out_dir, f"{prefix}_summary.csv")
    result.summary_df.to_csv(p1, index=False, encoding="utf-8-sig")
    paths["summary_csv"] = p1

    p2 = os.path.join(out_dir, f"{prefix}_field_scores.csv")
    result.field_scores_df.to_csv(p2, index=False, encoding="utf-8-sig")
    paths["field_scores_csv"] = p2

    p3 = os.path.join(out_dir, f"{prefix}_trial_scores.csv")
    result.trial_scores_df.to_csv(p3, index=False, encoding="utf-8-sig")
    paths["trial_scores_csv"] = p3

    p4 = os.path.join(out_dir, f"{prefix}_report.json")
    with open(p4, "w", encoding="utf-8") as f:
        json.dump(result.report, f, ensure_ascii=False, indent=2)
    paths["report_json"] = p4

    p5 = os.path.join(out_dir, f"{prefix}_uncovered_candidates.jsonl")
    _write_jsonl(p5, result.uncovered_candidates)
    paths["uncovered_candidates_jsonl"] = p5

    return paths


def build_tool_a_zip_bytes(
    result: ToolAResult,
    zip_name_prefix: str = "tool_a",
    include_uncovered_candidates: bool = True,
) -> bytes:
    """
    Streamlitの download_button 用：出力をZIPにして bytes で返す。
    - tool_a_summary.csv
    - tool_a_field_scores.csv
    - tool_a_trial_scores.csv
    - tool_a_report.json
    - (optional) tool_a_uncovered_candidates.jsonl
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{zip_name_prefix}_summary.csv",
            result.summary_df.to_csv(index=False, encoding="utf-8-sig"),
        )
        zf.writestr(
            f"{zip_name_prefix}_field_scores.csv",
            result.field_scores_df.to_csv(index=False, encoding="utf-8-sig"),
        )
        zf.writestr(
            f"{zip_name_prefix}_trial_scores.csv",
            result.trial_scores_df.to_csv(index=False, encoding="utf-8-sig"),
        )
        zf.writestr(
            f"{zip_name_prefix}_report.json",
            json.dumps(result.report, ensure_ascii=False, indent=2),
        )

        if include_uncovered_candidates:
            # jsonl
            lines = []
            for r in result.uncovered_candidates:
                lines.append(json.dumps(r, ensure_ascii=False))
            zf.writestr(
                f"{zip_name_prefix}_uncovered_candidates.jsonl",
                "\n".join(lines) + ("\n" if lines else ""),
            )

    return buf.getvalue()
