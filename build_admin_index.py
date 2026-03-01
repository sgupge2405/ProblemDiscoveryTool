from __future__ import annotations
import os
from pathlib import Path
import json
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import faiss  # pip install faiss-cpu

# -----------------------------------------
# パス・環境設定
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ★ コーパスは data/admin/ 配下
ADMIN_DOCS_DIR = BASE_DIR / "data" / "admin"

# ★ 管理者RAG用インデックス（admin_rag.py が読む）
INDEX_PATH = BASE_DIR / "admin_index.faiss"
META_PATH = BASE_DIR / "admin_meta.json"

# docx / pdf 読み込み用
try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def read_docx_file(path: Path) -> str:
    if docx is None:
        print("[WARN] python-docx が無いため docx をスキップします:", path)
        return ""
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)


def read_pdf_file(path: Path) -> str:
    if PyPDF2 is None:
        print("[WARN] PyPDF2 が無いため pdf をスキップします:", path)
        return ""

    text_parts: List[str] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(text_parts)


# ---- テキストをシンプルにチャンク分割 ----
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# ---- OpenAI APIで Embedding 作成（LangChainは使わない）----
def build_embeddings(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")


def main() -> None:
    if not ADMIN_DOCS_DIR.exists():
        raise RuntimeError(f"data/admin ディレクトリがありません: {ADMIN_DOCS_DIR}")

    # 対象は txt / md / docx / pdf
    targets = sorted(
        [
            p
            for p in ADMIN_DOCS_DIR.glob("**/*")
            if p.suffix.lower() in {".txt", ".md", ".docx", ".pdf"}
        ]
    )

    if not targets:
        raise RuntimeError(f"{ADMIN_DOCS_DIR} 配下に txt / md / docx / pdf がありません。")

    all_chunks: List[str] = []
    metas: List[Dict] = []

    print(f"[INFO] Loaded {len(targets)} documents")

    for path in targets:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            raw = read_text_file(path)
        elif suffix == ".docx":
            raw = read_docx_file(path)
        elif suffix == ".pdf":
            raw = read_pdf_file(path)
        else:
            continue

        if not raw.strip():
            continue

        chunks = chunk_text(raw, chunk_size=800, overlap=200)
        for idx, ch in enumerate(chunks):
            all_chunks.append(ch)
            metas.append(
                {
                    "source": str(path.relative_to(BASE_DIR)),
                    "chunk_index": idx,
                }
            )

    if not all_chunks:
        raise RuntimeError("チャンクが1件も生成されませんでした。ライブラリ不足等を確認してください。")

    print(f"[INFO] Split into {len(all_chunks)} chunks")

    # ---- Embedding 作成 ----
    vecs = build_embeddings(all_chunks)
    dim = vecs.shape[1]

    # ---- FAISS インデックス作成 ----
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))

    # ---- メタ情報を保存 ----
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "texts": all_chunks,
                "metas": metas,
                "embed_model": EMBED_MODEL,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] インデックス保存: {INDEX_PATH}")
    print(f"[INFO] メタ保存: {META_PATH}")


if __name__ == "__main__":
    main()
