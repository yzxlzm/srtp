from pathlib import Path
from typing import Iterable, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embeddings import get_embeddings


def build_vectorstore(docs: Iterable[Document], embeddings: Optional[object] = None) -> FAISS:
    """
    构建向量库，并输出简单的分批进度提示，方便观察耗时阶段。
    """
    embeddings = embeddings or get_embeddings()

    docs_list = list(docs)
    total = len(docs_list)
    if total == 0:
        raise ValueError("构建向量库失败：没有可用的文本块")

    print(f"开始构建向量库，共 {total} 个文本块...")
    batch_size = max(1, total // 10)  # 约 10 个进度节点

    # 先用第一批初始化，后续增量添加以展示进度
    first_batch = docs_list[:batch_size]
    store = FAISS.from_documents(first_batch, embeddings)
    processed = len(first_batch)
    print(f"进度: {processed}/{total} ({processed * 100 // total}%)")

    # 增量添加剩余文档
    while processed < total:
        next_batch = docs_list[processed : processed + batch_size]
        store.add_documents(next_batch)
        processed += len(next_batch)
        print(f"进度: {processed}/{total} ({processed * 100 // total}%)")

    print("向量库构建完成。")
    return store


def save_vectorstore(store: FAISS, path: str):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    store.save_local(path_obj)


def load_vectorstore(path: str, embeddings: Optional[object] = None) -> FAISS:
    embeddings = embeddings or get_embeddings()
    path_obj = Path(path)
    return FAISS.load_local(path_obj, embeddings, allow_dangerous_deserialization=True)

