from fastapi import FastAPI, Query as FastAPIQuery, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Callable
from sqlalchemy.ext.asyncio import AsyncSession
import sys
import os
from pathlib import Path
import torch
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline import build_rag_chain
from src.model_selector import ModelSelector
from config.DB_config import get_db
from crud.users import get_user_by_username, create_user, create_token, authenticate_users
from schemes.users import UserRequest
from utils.response import success_response
from utils.auth import get_current_user
from src.llm import get_llm,pre_load_qwen_lora
app = FastAPI()

# 先允许所有跨域，后续可根据需要调整为更严格的配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str
    model: Optional[str] = None  # 可选，指定要调用的大模型 key（见 llm.MODEL_REGISTRY）


_chain = None
_vectorstore = None


def get_chain():
    global _chain
    if _chain is None:
        _chain = build_rag_chain()
    return _chain


def get_vectorstore():
    """获取向量库对象（用于混合架构）"""
    global _vectorstore
    if _vectorstore is None:
        import os
        import tempfile
        from dotenv import load_dotenv
        from pathlib import Path
        from src.embeddings import get_embeddings
        from src.vectorstore import load_vectorstore, build_vectorstore, save_vectorstore
        from src.rag_pipeline import load_docs
        
        # 缓存目录
        load_dotenv()
        env_cache = os.getenv("VECTOR_CACHE_DIR")
        if env_cache:
            vector_dir = Path(env_cache)
        else:
            vector_dir = Path(tempfile.gettempdir()) / "vectorstore_cache"
        
        vector_dir.mkdir(parents=True, exist_ok=True)
        embeddings = get_embeddings()
        
        index_path = vector_dir
        index_f = index_path / "index.faiss"
        index_pkl = index_path / "index.pkl"
        
        try:
            if index_f.exists() and index_pkl.exists():
                _vectorstore = load_vectorstore(index_path, embeddings)
                print(f"已加载向量库：{index_path}")
            else:
                print("向量库不存在，开始构建...")
                docs = load_docs()
                _vectorstore = build_vectorstore(docs, embeddings)
                try:
                    save_vectorstore(_vectorstore, index_path)
                    print(f"向量库已构建并保存：{index_path}")
                except Exception as e:
                    print(f"向量库保存失败（继续使用内存向量库）：{e}")
        except Exception as e:
            print(f"加载向量库失败: {e}")
            _vectorstore = None
    
    return _vectorstore


_kg = None


_kg_cache = {}  # 支持缓存多个类型的知识图谱


def get_knowledge_graph(kg_type: str = "ml_full"):
    """
    获取知识图谱（懒加载）
    
    Args:
        kg_type: 知识图谱类型，"general" 或 "ml_full"
    """
    # 使用字典缓存，支持多个类型的知识图谱
    if kg_type not in _kg_cache:
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            import tempfile
            from pathlib import Path
            from src.knowledge_graph import KnowledgeGraph
            
            # 确定缓存路径
            env_cache = os.environ.get("VECTOR_CACHE_DIR")
            if env_cache:
                cache_dir = Path(env_cache).parent / "kg_cache"
            else:
                cache_dir = Path(tempfile.gettempdir()) / "kg_cache"
            
            if kg_type == "ml_full":
                kg_path = cache_dir / "ml_full_kg"
            else:
                kg_path = cache_dir / "knowledge_graph"
            
            # 尝试加载已有知识图谱
            kg = KnowledgeGraph()
            try:
                kg.load(kg_path)
                _kg_cache[kg_type] = kg
                print(f"✓ 已加载知识图谱（{kg_type}）：{kg_path}")
            except FileNotFoundError:
                # 如果不存在，尝试构建
                print(f"知识图谱不存在，开始构建（{kg_type}）...")
                if kg_type == "ml_full":
                    from src.knowledge_graph import build_ml_full_kg
                    kg = build_ml_full_kg(force_rebuild=False)
                else:
                    from src.knowledge_graph import build_knowledge_graph_from_docs
                    kg = build_knowledge_graph_from_docs()
                _kg_cache[kg_type] = kg
        except Exception as e:
            print(f"知识图谱加载失败（{kg_type}）: {e}")
            _kg_cache[kg_type] = None
    
    return _kg_cache.get(kg_type)

@app.on_event("startup")
async def preload_model():
    '''
    启动时预加载模型
    '''
    print("正在预加载本地模型...")
    try:
        # 现在的 get_llm 返回的是单个 LocalLLMWrapper 对象，直接获取即可，不需要解包
        _ = get_llm(model="qwen-lora")
        
        global local_model, tokenizer
        local_model, tokenizer = pre_load_qwen_lora()
        print("本地模型预加载完成")
    except Exception as e:
        print(f"本地模型预加载跳过或失败: {e}")

def get_llm_answer(model_key: Optional[str], answer_builder: Callable):
    """
    统一模型调用入口。
    由于 llm.py 已经将本地模型和 API 模型统一封装为了具备 .invoke() 的对象，
    这里无需再区分模型类型，直接获取并调用即可。

    Args:
        model_key: 模型 key（如 deepseek-chat / qwen-lora / openai-gpt-4o-mini）
        answer_builder: 接收 llm 并返回答案的函数
    """
    
    if model_key == 'qwen-lora':
        llm = get_llm(model=model_key, local_model=local_model, tokenizer=tokenizer)
    else:
        llm = get_llm(model=model_key)
    return answer_builder(llm)
   


@app.on_event("startup")
async def preload_chain():
    """启动时预先加载知识图谱和向量库，避免首个请求等待。"""
    print("正在初始化系统...")
    
    # 加载知识图谱
    print("1. 加载知识图谱...")
    kg = get_knowledge_graph(kg_type="ml_full")
    if kg:
        stats = kg.get_statistics()
        print(f"   ✓ 知识图谱加载完成：{stats['total_entities']} 个实体, {stats['total_relations']} 个关系")
    else:
        print("   ✗ 知识图谱加载失败，请先运行 python src/build_ml_kg.py 构建知识图谱")
    
    # 加载向量库
    print("2. 加载向量库...")
    vectorstore = get_vectorstore()
    if vectorstore:
        print(f"   ✓ 向量库加载完成")
    else:
        print("   ✗ 向量库加载失败")
    
    print("系统初始化完成")


@app.post("/register")
async def register(user: UserRequest, db: AsyncSession = Depends(get_db)):
    existing = await get_user_by_username(db, user.username)
    if existing:
        return {"code": 400, "message": "用户名已存在"}
    new_user = await create_user(db, user)
    token = await create_token(db, new_user.id)
    return success_response("注册成功", {"token": token})


@app.post("/login")
async def login(user: UserRequest, db: AsyncSession = Depends(get_db)):
    auth_user = await authenticate_users(db, user.username, user.password)
    if not auth_user:
        return {"code": 401, "message": "用户名或密码错误"}
    token = await create_token(db, auth_user.id)
    return success_response("登录成功", {"token": token})


@app.post("/query")
async def query(
    q: Query,
    kg_type: Optional[str] = FastAPIQuery("ml_full", description="知识图谱类型: general 或 ml_full"),
    current_user = Depends(get_current_user)
):
    """
    主要查询接口 - 混合架构（知识图谱 + RAG）
    
    流程：
    1. LLM理解问题
    2. 知识图谱定位相关知识
    3. RAG检索教材内容
    4. LLM生成结构化回答
    
    Args:
        q: 用户问题
        kg_type: 知识图谱类型，"general" 或 "ml_full"（默认ml_full）
    """
    kg = get_knowledge_graph(kg_type=kg_type)
    if kg is None:
        return {
            "answer": "知识图谱未构建，请先调用 /kg/build?kg_type=" + kg_type + " 构建知识图谱"
        }
    
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return {
            "answer": "向量库未构建，请先启动服务以自动构建向量库"
        }
    
    try:
        from src.knowledge_graph import hybrid_kg_rag_answer

        # 自动选择模型
        if q.model and q.model != "auto":
            selected_model = q.model
        else:
            # 只有前端选了“自动选择(auto)”，才让路由器介入
            selected_model = ModelSelector.select_model(q.question, q.model)
        answer = get_llm_answer(
            selected_model,
            lambda llm: hybrid_kg_rag_answer(q.question, kg, vectorstore, llm=llm),
        )
        return {"answer": answer, "model_used": selected_model}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"生成回答时出错: {str(e)}"
        }


@app.post("/kg/build")
async def build_kg(kg_type: Optional[str] = FastAPIQuery("ml_full", description="知识图谱类型: general 或 ml_full")):
    """
    构建知识图谱
    
    Args:
        kg_type: 知识图谱类型，"general" 或 "ml_full"
    """
    try:
        if kg_type == "ml_full":
            from src.knowledge_graph import build_ml_full_kg
            kg = build_ml_full_kg(force_rebuild=True)
        else:
            from src.knowledge_graph import build_knowledge_graph_from_docs
            kg = build_knowledge_graph_from_docs()
        stats = kg.get_statistics()
        return {
            "status": "success",
            "message": "知识图谱构建完成",
            "statistics": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"构建知识图谱失败: {str(e)}"
        }


@app.post("/neo4j/sync")
async def neo4j_sync(kg_type: Optional[str] = FastAPIQuery("ml_full", description="知识图谱类型: general 或 ml_full")):
    """
    将本地缓存/构建的 KnowledgeGraph 同步到 Neo4j（用于 GraphRAG）。
    需要在 .env 中配置 NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD/NEO4J_DATABASE。
    """
    kg = get_knowledge_graph(kg_type=kg_type)
    if kg is None:
        return {"status": "error", "message": f"知识图谱未构建，请先调用 /kg/build?kg_type={kg_type}"}

    try:
        from neo4j_store import Neo4jStore

        store = Neo4jStore()
        try:
            store.ensure_schema()
            store.upsert_entities(kg.entities)
            store.upsert_relations(kg.relations)
            return {"status": "success", "message": "已同步到 Neo4j", "statistics": store.stats()}
        finally:
            store.close()
    except Exception as e:
        return {"status": "error", "message": f"同步到 Neo4j 失败: {str(e)}"}


@app.post("/graphrag/query")
async def graphrag_query(q: Query, kg_type: Optional[str] = FastAPIQuery("ml_full", description="知识图谱类型: general 或 ml_full")):
    """
    GraphRAG（简化版）：LLM 提取关键概念 -> Neo4j 拉取相关子图 -> LLM 基于子图上下文回答。
    说明：这是“可跑通”的最小闭环，后续可升级为全文/向量检索 + 更复杂的子图选择策略。
    """
    try:
        from neo4j_store import Neo4jStore
        from src.knowledge_graph import understand_question
        from langchain_core.prompts import ChatPromptTemplate

        # 1) 抽取关键概念（复用现有逻辑）
        selected_model = q.model or "deepseek-chat"
        analysis = get_llm_answer(
            selected_model,
            lambda llm: understand_question(q.question, llm=llm),
        )
        key_concepts = analysis.get("key_concepts", []) or [q.question]

        # 2) Neo4j 侧：实体匹配 + 拉子图三元组
        store = Neo4jStore()
        try:
            matched = []
            for c in key_concepts[:5]:
                matched.extend([x["name"] for x in store.search_entities(c, limit=5)])
            matched = list(dict.fromkeys(matched))[:10]
            triples = store.fetch_subgraph_by_entities(matched, hops=1, limit_rels=80)
        finally:
            store.close()

        triples_text = "\n".join([f"- {h} --[{r}]--> {t}" for (h, r, t) in triples]) if triples else ""
        entities_text = ", ".join(matched) if matched else ""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的机器学习知识助手。请基于给定的“图谱三元组”回答用户问题。

    要求：
    1. 直接回答问题，给出清晰、有条理的解释。
    2. 优先使用三元组信息组织答案；如果三元组不足以覆盖问题，可用通用知识补充，但不要提及“图谱/三元组/上下文”等字眼。
    3. 若涉及公式，可用 LaTeX（$...$ 或 $$...$$）表达。""",
                ),
                ("human", "问题：{question}\n\n命中的实体：{entities}\n\n相关三元组：\n{triples}"),
            ]
        )

        answer = get_llm_answer(
            selected_model,
            lambda llm: llm.invoke(
                prompt.format_messages(
                    question=q.question,
                    entities=entities_text,
                    triples=triples_text,
                )
            ).content.strip(),
        )
        return {
            "status": "success",
            "model": selected_model,
            "key_concepts": key_concepts,
            "matched_entities": matched,
            "triples_count": len(triples),
            "answer": answer,
        }
    except Exception as e:
        return {"status": "error", "message": f"GraphRAG 查询失败: {str(e)}"}


@app.get("/kg/stats")
async def kg_stats():
    """获取知识图谱统计信息"""
    kg = get_knowledge_graph()
    if kg is None:
        return {
            "status": "error",
            "message": "知识图谱未构建，请先调用 /kg/build"
        }
    return {
        "status": "success",
        "statistics": kg.get_statistics()
    }


@app.post("/kg/query")
async def query_kg(q: Query, kg_type: Optional[str] = FastAPIQuery("general", description="知识图谱类型: general 或 ml_full")):
    """
    从知识图谱中查询
    
    Args:
        q: 查询问题
        kg_type: 知识图谱类型，"general" 或 "ml_full"
    """
    kg = get_knowledge_graph(kg_type=kg_type)
    if kg is None:
        return {
            "status": "error",
            "message": "知识图谱未构建，请先调用 /kg/build"
        }
    
    try:
        from src.knowledge_graph import query_knowledge_graph, format_query_results
        results = query_knowledge_graph(kg, q.question)
        formatted = format_query_results(results, max_results=20)
        return {
            "status": "success",
            "query": q.question,
            "results": results,
            "formatted": formatted
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"查询失败: {str(e)}"
        }


@app.post("/kg/generate")
async def generate_from_kg(q: Query, kg_type: Optional[str] = FastAPIQuery("ml_full", description="知识图谱类型: general 或 ml_full")):
    """
    基于知识图谱生成回答（与 /query 功能相同，返回格式不同）
    
    Args:
        q: 用户问题
        kg_type: 知识图谱类型，"general" 或 "ml_full"
    """
    kg = get_knowledge_graph(kg_type=kg_type)
    if kg is None:
        return {
            "status": "error",
            "message": "知识图谱未构建，请先调用 /kg/build?kg_type=" + kg_type
        }
    
    try:
        from src.knowledge_graph import generate_answer_from_kg
        answer = generate_answer_from_kg(kg, q.question)
        return {
            "status": "success",
            "question": q.question,
            "answer": answer
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"生成回答失败: {str(e)}"
        }


@app.post("/query/vector")  # 保留向量库RAG作为备用接口
async def query_vector(q: Query):
    """
    使用向量库RAG进行回答（备用接口）
    
    Args:
        q: 用户问题
    """
    chain = get_chain()
    res = chain.invoke({"question": q.question})
    return {"answer": res.content}


@app.get("/", response_class=HTMLResponse)
async def index():
    from pathlib import Path
    html_path = Path(__file__).parent.parent / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

