from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """使用更轻量的 BGE（bge-small）以提升构建速度，强制在 CPU 上运行。"""
    model_name = "BAAI/bge-small-zh-v1.5"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )

