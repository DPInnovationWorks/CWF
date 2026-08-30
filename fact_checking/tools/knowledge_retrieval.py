"""
知识库检索工具：从本地文档目录构建向量检索，供 Agent 调用。
默认使用本地 HuggingFace 嵌入模型。
"""

from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool

from config.env import ensure_loaded


def build_knowledge_retriever_tool(
    knowledge_base_path: str | Path,
    persist_directory: str | Path,
    *,
    tool_name: str = "knowledge_base_search",
    tool_description: str = "从知识库中检索与问题相关的文档片段。用于回答产品说明、内部文档、FAQ 等。",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    k: int = 4,
    use_openai_embedding: bool = False,
    glob: str = "**/*.txt",
) -> List:
    """
    从指定目录加载文档，构建向量检索器并封装为 LangChain Tool。
    返回包含 1 个 retrieval tool 的列表；若路径不存在或无文档则返回空列表。

    :param knowledge_base_path: 存放 txt/md 等文档的目录
    :param persist_directory: Chroma 持久化目录（同一知识库可复用以加速）
    :param tool_name: 工具名称（给 LLM 用）
    :param tool_description: 工具描述
    :param chunk_size: 文本切分块大小
    :param chunk_overlap: 块重叠长度
    :param k: 检索返回条数
    :param use_openai_embedding: 是否使用 OpenAI 嵌入（需 OPENAI_API_KEY）
    :param glob: 匹配文件模式，如 "**/*.txt" 或 "**/*.md"
    """
    ensure_loaded()
    path = Path(knowledge_base_path)
    if not path.exists() or not path.is_dir():
        return []

    # 嵌入模型
    if use_openai_embedding:
        from langchain_openai import OpenAIEmbeddings

        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    persist_dir = Path(persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 若已有持久化且目录未变，可直接加载
    chroma_index = persist_dir / "chroma"
    if chroma_index.exists():
        try:
            from langchain_chroma import Chroma

            vector_store = Chroma(
                persist_directory=str(chroma_index),
                embedding_function=embedding,
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": k})
            tool = create_retriever_tool(
                retriever,
                name=tool_name,
                description=tool_description,
            )
            return [tool]
        except Exception:
            pass

    # 加载文档
    loader = DirectoryLoader(
        str(path),
        glob=glob,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    from langchain_chroma import Chroma

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(chroma_index),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    tool = create_retriever_tool(
        retriever,
        name=tool_name,
        description=tool_description,
    )
    return [tool]


def get_default_knowledge_tools(
    knowledge_base_path: Optional[str | Path] = None,
    persist_directory: Optional[str | Path] = None,
) -> List:
    """
    使用项目内默认路径构建知识库工具。
    若未传参，则使用项目根目录下的 data/knowledge 与 data/knowledge_index。
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    base = knowledge_base_path or root / "data" / "knowledge"
    persist = persist_directory or root / "data" / "knowledge_index"
    return build_knowledge_retriever_tool(base, persist)
