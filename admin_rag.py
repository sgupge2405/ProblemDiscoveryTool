# admin_rag.py
from __future__ import annotations
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

ADMIN_INDEX_DIR = "indexes/faiss_admin"

_vectordb: Optional[FAISS] = None


def _get_vectordb() -> Optional[FAISS]:
    """管理者ファイル用ベクトルDBを遅延ロードする"""
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    if not os.path.exists(ADMIN_INDEX_DIR):
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _vectordb = FAISS.load_local(
        ADMIN_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return _vectordb


def get_admin_hints(query: str, k: int = 4, max_chars: int = 1200) -> str:
    """
    管理者RAGから「観点用のヒントテキスト」を取得する。
    - ユーザーにそのまま見せる前提ではなく、LLMへの内部ヒントとしてのみ使う。
    - 長くなりすぎないよう max_chars でカットする。
    """
    vectordb = _get_vectordb()
    if vectordb is None:
        return ""

    docs = vectordb.similarity_search(query, k=k)
    if not docs:
        return ""

    joined = "\n\n".join(d.page_content for d in docs)
    return joined[:max_chars]
