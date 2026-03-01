import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

ADMIN_DOCS_DIR = Path("data/admin")
ADMIN_INDEX_DIR = Path("indexes/faiss_admin")

def load_text_docs():
    loaders = []
    for p in sorted(list(ADMIN_DOCS_DIR.rglob("*.md")) + list(ADMIN_DOCS_DIR.rglob("*.txt"))):
        loaders.append(TextLoader(str(p), encoding="utf-8"))
    docs = []
    for ld in loaders:
        docs.extend(ld.load())
    return docs

def main():
    if not ADMIN_DOCS_DIR.exists():
        raise SystemExit(f"Not found: {ADMIN_DOCS_DIR.resolve()}")

    docs = load_text_docs()
    if not docs:
        raise SystemExit("No .md/.txt files found under data/admin")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(splits, embeddings)

    os.makedirs(ADMIN_INDEX_DIR, exist_ok=True)
    vectordb.save_local(str(ADMIN_INDEX_DIR))

    print("OK")
    print(f"Saved: {ADMIN_INDEX_DIR}")
    print(f"Docs: {len(docs)}  Chunks: {len(splits)}")

if __name__ == "__main__":
    main()
