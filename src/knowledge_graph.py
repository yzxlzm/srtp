"""
知识图谱构建模块
从文档中提取实体和关系，构建知识图谱
"""
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import tempfile

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from llm import get_llm
from rag_pipeline import load_docs, load_docx
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeGraph:
    """知识图谱类，用于存储和管理实体、关系"""
    
    def __init__(self):
        self.entities: Dict[str, Dict] = {}  # 实体: {name: {type, properties}}
        self.relations: List[Dict] = []  # 关系: [{head, relation, tail, source}]
        self.graph = None
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
    
    def add_entity(self, name: str, entity_type: str, properties: Optional[Dict] = None):
        """添加实体"""
        if name not in self.entities:
            self.entities[name] = {
                "type": entity_type,
                "properties": properties or {}
            }
            if self.graph:
                # 避免 properties 中包含 "type" 导致与显式参数冲突
                props = dict(properties or {})
                props.pop("type", None)
                self.graph.add_node(name, type=entity_type, **props)
    
    def add_relation(self, head: str, relation: str, tail: str, source: Optional[str] = None):
        """添加关系"""
        # 确保实体存在
        if head not in self.entities:
            self.add_entity(head, "Unknown")
        if tail not in self.entities:
            self.add_entity(tail, "Unknown")
        
        relation_data = {
            "head": head,
            "relation": relation,
            "tail": tail,
            "source": source
        }
        
        # 避免重复关系
        if relation_data not in self.relations:
            self.relations.append(relation_data)
            if self.graph:
                self.graph.add_edge(head, tail, relation=relation, source=source)
    
    def get_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        entity_types = {}
        for entity in self.entities.values():
            entity_type = entity.get("type", "Unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        relation_types = {}
        for rel in self.relations:
            rel_type = rel.get("relation", "Unknown")
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
            
            
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "graph_nodes": len(self.graph.nodes()) if self.graph else 0,
            "graph_edges": len(self.graph.edges()) if self.graph else 0
        }
    
    def save(self, path: Path):
        """保存知识图谱到文件"""
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存为 JSON
        data = {
            "entities": self.entities,
            "relations": self.relations
        }
        json_path = path / "knowledge_graph.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 保存 NetworkX 图（如果可用）
        if self.graph and HAS_NETWORKX:
            graph_path = path / "knowledge_graph.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(self.graph, f)
        
        print(f"知识图谱已保存到: {path}")
    
    def load(self, path: Path):
        """从文件加载知识图谱"""
        json_path = path / "knowledge_graph.json"
        if not json_path.exists():
            raise FileNotFoundError(f"知识图谱文件不存在: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.entities = data.get("entities", {})
        self.relations = data.get("relations", [])
        
        # 重建 NetworkX 图
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
            for name, entity in self.entities.items():
                # 避免 properties 中包含 "type" 导致与显式参数冲突
                props = dict(entity.get("properties", {}) or {})
                props.pop("type", None)
                self.graph.add_node(name, type=entity.get("type"), **props)
            for rel in self.relations:
                self.graph.add_edge(
                    rel["head"], 
                    rel["tail"], 
                    relation=rel["relation"], 
                    source=rel.get("source")
                )
        
        print(f"知识图谱已加载: {len(self.entities)} 个实体, {len(self.relations)} 个关系")


def extract_entities_relations(text: str, llm, domain: str = "general") -> Tuple[List[Dict], List[Dict]]:
    """
    使用 LLM 从文本中提取实体和关系
    返回: (entities, relations)
    
    Args:
        text: 要提取的文本
        llm: LLM 模型
        domain: 领域类型，"general" 或 "machine_learning"
    """
    if domain == "machine_learning":
        prompt_template = """你是一个机器学习知识图谱构建专家。请从给定的文本中提取实体和关系。

实体类型包括但不限于：
- 概念/术语：如"监督学习"、"过拟合"、"交叉验证"等
- 算法/方法：如"决策树"、"支持向量机"、"神经网络"等
- 模型：如"线性回归"、"逻辑回归"、"随机森林"等
- 技术/技术点：如"特征选择"、"正则化"、"集成学习"等
- 数据集/数据：如"训练集"、"测试集"、"验证集"等
- 评估指标：如"准确率"、"精确率"、"召回率"、"F1分数"等
- 问题/任务：如"分类"、"回归"、"聚类"等
- 属性/特征：如"特征向量"、"标签"、"样本"等
- 理论/原理：如"偏差-方差权衡"、"奥卡姆剃刀"等

关系类型包括但不限于：
- 属于/是：表示分类关系，如"决策树属于监督学习"
- 用于/应用于：表示应用关系，如"交叉验证用于模型选择"
- 解决/处理：表示解决问题，如"正则化解决过拟合问题"
- 评估/衡量：表示评估关系，如"准确率评估分类性能"
- 包含/包括：表示包含关系，如"训练集包含多个样本"
- 基于/依赖：表示依赖关系，如"集成学习基于多个基学习器"
- 优于/胜过：表示比较关系，如"随机森林优于单一决策树"
- 导致/引起：表示因果关系，如"过拟合导致泛化能力差"
- 需要/要求：表示需求关系，如"监督学习需要标注数据"
- 定义/描述：表示定义关系，如"准确率定义为正确预测的比例"

请以 JSON 格式返回结果，格式如下：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "properties": {{"key": "value"}}}}
    ],
    "relations": [
        {{"head": "头实体", "relation": "关系类型", "tail": "尾实体"}}
    ]
}}

只返回 JSON，不要其他文字。"""
    else:
        prompt_template = """你是一个知识图谱构建专家。请从给定的文本中提取实体和关系。

实体类型包括但不限于：
- 学校/高校
- 专业/学科
- 企业/公司
- 职位/岗位
- 毕业生/学生
- 地区/城市
- 行业
- 技能/能力

关系类型包括但不限于：
- 毕业于
- 位于
- 属于
- 需要
- 掌握
- 从事

请以 JSON 格式返回结果，格式如下：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "properties": {{"key": "value"}}}}
    ],
    "relations": [
        {{"head": "头实体", "relation": "关系类型", "tail": "尾实体"}}
    ]
}}

只返回 JSON，不要其他文字。"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "文本内容：\n{text}")
    ])
    
    try:
        response = llm.invoke(prompt.format_messages(text=text))
        content = response.content.strip()
        
        # 尝试提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        entities = result.get("entities", [])
        relations = result.get("relations", [])
        
        return entities, relations
    except Exception as e:
        print(f"提取实体和关系时出错: {e}")
        print(f"响应内容: {content[:200]}...")  # 打印前200个字符用于调试
        return [], []


def build_knowledge_graph(docs: List[Document], llm, batch_size: int = 5, domain: str = "general") -> KnowledgeGraph:
    """
    从文档中构建知识图谱
    
    Args:
        docs: 文档列表
        llm: LLM 模型
        batch_size: 批处理大小
        domain: 领域类型，"general" 或 "machine_learning"
    """
    if not HAS_NETWORKX:
        print("警告: 未安装 networkx，知识图谱功能受限。请运行: pip install networkx")
    
    kg = KnowledgeGraph()
    total = len(docs)
    
    print(f"开始构建知识图谱，共 {total} 个文档...")
    
    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        batch_text = "\n\n".join([doc.page_content for doc in batch])
        
        # 提取实体和关系
        entities, relations = extract_entities_relations(batch_text, llm, domain=domain)
        
        # 添加到知识图谱
        for entity in entities:
            kg.add_entity(
                name=entity.get("name", ""),
                entity_type=entity.get("type", "Unknown"),
                properties=entity.get("properties", {})
            )
        
        for relation in relations:
            source = batch[0].metadata.get("source", "unknown") if batch else None
            kg.add_relation(
                head=relation.get("head", ""),
                relation=relation.get("relation", ""),
                tail=relation.get("tail", ""),
                source=source
            )
        
        processed = min(i + batch_size, total)
        print(f"进度: {processed}/{total} ({processed * 100 // total}%)")
    
    print("知识图谱构建完成。")
    return kg


def build_knowledge_graph_from_docs() -> KnowledgeGraph:
    """
    从 data 目录的文档构建知识图谱
    """
    base_dir = Path(__file__).resolve().parent.parent
    
    # 缓存目录
    env_cache = os.environ.get("VECTOR_CACHE_DIR")
    if env_cache:
        cache_dir = Path(env_cache).parent / "kg_cache"
    else:
        cache_dir = Path(tempfile.gettempdir()) / "kg_cache"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    kg_path = cache_dir / "knowledge_graph"
    
    # 尝试加载已有知识图谱
    try:
        if (kg_path / "knowledge_graph.json").exists():
            kg = KnowledgeGraph()
            kg.load(kg_path)
            print(f"已加载缓存知识图谱：{kg_path}")
            return kg
    except Exception as e:
        print(f"加载知识图谱失败，将重新构建: {e}")
    
    # 构建新知识图谱
    print("开始从文档构建知识图谱...")
    docs = load_docs()
    llm = get_llm()
    
    kg = build_knowledge_graph(docs, llm, batch_size=5)
    
    # 保存知识图谱
    try:
        kg.save(kg_path)
    except Exception as e:
        print(f"保存知识图谱失败: {e}")
    
    return kg


def build_ml_full_kg(docx_filename: str = "【电子书】周志华-机器学习.docx",
                     force_rebuild: bool = False) -> KnowledgeGraph:
    """
    从机器学习文档的全量内容构建知识图谱
    
    Args:
        docx_filename: docx文件名
        force_rebuild: 是否强制重建（忽略缓存）
        
    Returns:
        知识图谱对象
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    docx_path = data_dir / docx_filename
    
    if not docx_path.exists():
        raise FileNotFoundError(f"未找到文档: {docx_path}")
    
    # 缓存目录
    env_cache = os.environ.get("VECTOR_CACHE_DIR")
    if env_cache:
        cache_dir = Path(env_cache).parent / "kg_cache"
    else:
        cache_dir = Path(tempfile.gettempdir()) / "kg_cache"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    kg_path = cache_dir / "ml_full_kg"
    
    # 尝试加载已有知识图谱
    if not force_rebuild:
        try:
            if (kg_path / "knowledge_graph.json").exists():
                kg = KnowledgeGraph()
                kg.load(kg_path)
                print(f"已加载缓存知识图谱：{kg_path}")
                return kg
        except Exception as e:
            print(f"加载知识图谱失败，将重新构建: {e}")
    
    print(f"开始构建全量机器学习知识图谱：{docx_path}")
    # 全量读取 docx
    full_docs = load_docx(docx_path)
    if not full_docs:
        raise ValueError("未能从文档中提取内容")
    
    # 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(full_docs)
    print(f"文档分割完成，共生成 {len(split_docs)} 个文本块")
    
    # 构建知识图谱
    llm = get_llm()
    kg = build_knowledge_graph(split_docs, llm, batch_size=3, domain="machine_learning")
    
    # 保存
    try:
        kg.save(kg_path)
        print(f"知识图谱已保存到: {kg_path}")
    except Exception as e:
        print(f"保存知识图谱失败: {e}")
    
    # 打印统计信息
    stats = kg.get_statistics()
    print("\n" + "="*50)
    print("知识图谱统计信息:")
    print(f"  实体总数: {stats['total_entities']}")
    print(f"  关系总数: {stats['total_relations']}")
    print(f"  实体类型分布:")
    for entity_type, count in stats['entity_types'].items():
        print(f"    - {entity_type}: {count}")
    print(f"  关系类型分布:")
    for relation_type, count in stats['relation_types'].items():
        print(f"    - {relation_type}: {count}")
    print("="*50)
    
    return kg


def query_knowledge_graph(kg: KnowledgeGraph, question: str) -> List[Dict]:
    """
    从知识图谱中查询相关信息（关键词匹配）
    
    Args:
        kg: 知识图谱对象
        question: 查询问题或关键词
        
    Returns:
        查询结果列表
    """
    results = []
    
    # 简单的关键词匹配查询
    question_lower = question.lower()
    
    # 查找相关实体
    for entity_name, entity_data in kg.entities.items():
        if question_lower in entity_name.lower() or entity_name.lower() in question_lower:
            results.append({
                "type": "entity",
                "name": entity_name,
                "data": entity_data
            })
    
    # 查找相关关系
    for relation in kg.relations:
        if (question_lower in relation["head"].lower() or 
            question_lower in relation["tail"].lower() or
            question_lower in relation["relation"].lower()):
            results.append({
                "type": "relation",
                "data": relation
            })
    
    return results


def query_entity_by_name(kg: KnowledgeGraph, entity_name: str) -> Optional[Dict]:
    """
    根据实体名称精确查询实体
    
    Args:
        kg: 知识图谱对象
        entity_name: 实体名称
        
    Returns:
        实体信息，如果不存在返回None
    """
    if entity_name in kg.entities:
        return {
            "name": entity_name,
            "data": kg.entities[entity_name]
        }
    return None


def query_relations_by_entity(kg: KnowledgeGraph, entity_name: str) -> List[Dict]:
    """
    查询与指定实体相关的所有关系
    
    Args:
        kg: 知识图谱对象
        entity_name: 实体名称
        
    Returns:
        相关关系列表
    """
    results = []
    entity_lower = entity_name.lower()
    
    for relation in kg.relations:
        if (relation["head"].lower() == entity_lower or 
            relation["tail"].lower() == entity_lower):
            results.append(relation)
    
    return results


def query_entities_by_type(kg: KnowledgeGraph, entity_type: str) -> List[Dict]:
    """
    根据实体类型查询所有实体
    
    Args:
        kg: 知识图谱对象
        entity_type: 实体类型
        
    Returns:
        该类型的所有实体列表
    """
    results = []
    entity_type_lower = entity_type.lower()
    
    for name, data in kg.entities.items():
        if data.get("type", "").lower() == entity_type_lower:
            results.append({
                "name": name,
                "data": data
            })
    
    return results


def query_relations_by_type(kg: KnowledgeGraph, relation_type: str) -> List[Dict]:
    """
    根据关系类型查询所有关系
    
    Args:
        kg: 知识图谱对象
        relation_type: 关系类型
        
    Returns:
        该类型的所有关系列表
    """
    results = []
    relation_type_lower = relation_type.lower()
    
    for relation in kg.relations:
        if relation.get("relation", "").lower() == relation_type_lower:
            results.append(relation)
    
    return results


def get_entity_neighbors(kg: KnowledgeGraph, entity_name: str, max_depth: int = 1) -> Dict:
    """
    获取实体的邻居节点（通过关系连接的实体）
    
    Args:
        kg: 知识图谱对象
        entity_name: 实体名称
        max_depth: 最大深度（暂时只支持1）
        
    Returns:
        包含邻居实体和关系的字典
    """
    neighbors = {
        "entity": entity_name,
        "outgoing": [],  # 作为头实体的关系
        "incoming": [],  # 作为尾实体的关系
        "connected_entities": set()
    }
    
    entity_lower = entity_name.lower()
    
    for relation in kg.relations:
        if relation["head"].lower() == entity_lower:
            neighbors["outgoing"].append(relation)
            neighbors["connected_entities"].add(relation["tail"])
        elif relation["tail"].lower() == entity_lower:
            neighbors["incoming"].append(relation)
            neighbors["connected_entities"].add(relation["head"])
    
    neighbors["connected_entities"] = list(neighbors["connected_entities"])
    
    return neighbors


def format_query_results(results: List[Dict], max_results: int = 20) -> str:
    """
    格式化查询结果为可读字符串
    
    Args:
        results: 查询结果列表
        max_results: 最大显示结果数
        
    Returns:
        格式化后的字符串
    """
    if not results:
        return "未找到相关结果。"
    
    output = []
    output.append(f"找到 {len(results)} 个结果（显示前 {min(len(results), max_results)} 个）：\n")
    
    entities = [r for r in results if r.get("type") == "entity"][:max_results]
    relations = [r for r in results if r.get("type") == "relation"][:max_results]
    
    if entities:
        output.append("【实体】")
        for i, item in enumerate(entities, 1):
            name = item.get("name", "未知")
            data = item.get("data", {})
            entity_type = data.get("type", "未知类型")
            output.append(f"  {i}. {name} ({entity_type})")
        output.append("")
    
    if relations:
        output.append("【关系】")
        for i, item in enumerate(relations, 1):
            rel_data = item.get("data", {})
            head = rel_data.get("head", "未知")
            relation = rel_data.get("relation", "未知关系")
            tail = rel_data.get("tail", "未知")
            output.append(f"  {i}. {head} --[{relation}]--> {tail}")
        output.append("")
    
    return "\n".join(output)


def extract_kg_context(kg: KnowledgeGraph, question: str, max_entities: int = 10, max_relations: int = 15) -> str:
    """
    从知识图谱中提取与问题相关的上下文信息
    
    Args:
        kg: 知识图谱对象
        question: 用户问题
        max_entities: 最大实体数量
        max_relations: 最大关系数量
        
    Returns:
        格式化的上下文字符串
    """
    # 查询相关实体和关系
    results = query_knowledge_graph(kg, question)
    
    if not results:
        return ""
    
    # 提取实体和关系
    entities = [r for r in results if r.get("type") == "entity"][:max_entities]
    relations = [r for r in results if r.get("type") == "relation"][:max_relations]
    
    # 如果找到实体，扩展查询其邻居关系
    if entities:
        entity_names = [e.get("name") for e in entities[:5]]  # 只扩展前5个实体的邻居
        for entity_name in entity_names:
            try:
                neighbors = get_entity_neighbors(kg, entity_name)
                # 添加邻居关系
                relations.extend(neighbors["outgoing"][:3])
                relations.extend(neighbors["incoming"][:3])
            except:
                pass  # 如果查询失败，继续处理其他实体
    
    # 去重关系
    seen_relations = set()
    unique_relations = []
    for rel in relations:
        # 处理不同格式的关系数据
        if isinstance(rel, dict):
            if "data" in rel:
                rel_data = rel["data"]
            else:
                rel_data = rel
        else:
            continue
        
        rel_key = (rel_data.get("head", ""), rel_data.get("relation", ""), rel_data.get("tail", ""))
        if rel_key and rel_key not in seen_relations:
            seen_relations.add(rel_key)
            unique_relations.append(rel_data)
    
    # 构建上下文
    context_parts = []
    
    if entities:
        context_parts.append("相关实体：")
        for entity in entities[:max_entities]:
            name = entity.get("name", "未知")
            data = entity.get("data", {})
            entity_type = data.get("type", "未知类型")
            context_parts.append(f"  - {name}（类型：{entity_type}）")
        context_parts.append("")
    
    if unique_relations:
        context_parts.append("相关关系：")
        for rel in unique_relations[:max_relations]:
            head = rel.get("head", "未知")
            relation = rel.get("relation", "未知关系")
            tail = rel.get("tail", "未知")
            context_parts.append(f"  - {head} --[{relation}]--> {tail}")
        context_parts.append("")
    
    return "\n".join(context_parts)


def generate_answer_from_kg(kg: KnowledgeGraph, question: str, llm=None) -> str:
    """
    基于知识图谱生成回答
    
    Args:
        kg: 知识图谱对象
        question: 用户问题
        llm: LLM模型（如果为None则自动获取）
        
    Returns:
        生成的回答
    """
    if llm is None:
        llm = get_llm()
    
    # 从知识图谱中提取相关上下文
    kg_context = extract_kg_context(kg, question)
    
    if not kg_context:
        # 如果没有找到相关知识，使用通用回答
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一个专业的机器学习知识助手。请基于你的知识回答用户的问题。"
            ),
            ("human", "问题：{question}")
        ])
    else:
        # 基于知识图谱上下文生成回答
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个专业的机器学习知识助手。请基于提供的知识图谱信息回答用户的问题。

重要规则：
1. 优先使用知识图谱中的信息来回答问题。
2. 如果知识图谱中的信息能够回答用户的问题，请直接基于这些信息给出准确、专业的回答。
3. 如果知识图谱中的信息不足，可以结合你的知识进行补充，但要明确说明哪些信息来自知识图谱。
4. 回答要专业、清晰、有条理。
5. 不要提及"知识图谱"、"上下文"等术语，直接给出答案。

知识图谱信息：
{kg_context}"""
            ),
            ("human", "问题：{question}")
        ])
    
    try:
        if kg_context:
            response = llm.invoke(prompt.format_messages(question=question, kg_context=kg_context))
        else:
            response = llm.invoke(prompt.format_messages(question=question))
        return response.content.strip()
    except Exception as e:
        return f"生成回答时出错: {e}"


def understand_question(question: str, llm=None) -> Dict:
    """
    使用LLM理解问题，提取关键概念和查询意图
    
    Args:
        question: 用户问题
        llm: LLM模型（如果为None则自动获取）
        
    Returns:
        包含关键概念、查询类型、搜索关键词的字典
    """
    if llm is None:
        llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个问题分析专家。请分析用户的问题，提取关键信息。

请以 JSON 格式返回结果，格式如下：
{{
    "key_concepts": ["概念1", "概念2", ...],  // 问题中的关键概念
    "query_type": "类型",  // 问题类型：定义、比较、应用、原理等
    "search_keywords": ["关键词1", "关键词2", ...],  // 用于检索的关键词
    "intent": "查询意图的简要描述"
}}

只返回 JSON，不要其他文字。"""
        ),
        ("human", "问题：{question}")
    ])
    
    try:
        response = llm.invoke(prompt.format_messages(question=question))
        content = response.content.strip()
        
        # 尝试提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"理解问题时出错: {e}")
        # 返回默认值
        return {
            "key_concepts": [question],
            "query_type": "未知",
            "search_keywords": [question],
            "intent": "理解问题"
        }


def locate_knowledge_in_kg(kg: KnowledgeGraph, key_concepts: List[str], max_entities: int = 15, max_relations: int = 20) -> Dict:
    """
    在知识图谱中定位相关知识
    
    Args:
        kg: 知识图谱对象
        key_concepts: 关键概念列表
        max_entities: 最大实体数量
        max_relations: 最大关系数量
        
    Returns:
        包含相关实体、关系、实体名称列表的字典
    """
    all_entities = []
    all_relations = []
    entity_names = set()
    
    # 对每个关键概念进行查询
    for concept in key_concepts:
        results = query_knowledge_graph(kg, concept)
        
        # 提取实体
        entities = [r for r in results if r.get("type") == "entity"]
        for entity in entities[:max_entities // len(key_concepts) if key_concepts else max_entities]:
            if entity not in all_entities:
                all_entities.append(entity)
                entity_name = entity.get("name", "")
                if entity_name:
                    entity_names.add(entity_name)
        
        # 提取关系
        relations = [r for r in results if r.get("type") == "relation"]
        for relation in relations[:max_relations // len(key_concepts) if key_concepts else max_relations]:
            rel_data = relation.get("data", relation) if isinstance(relation, dict) and "data" in relation else relation
            if rel_data not in all_relations:
                all_relations.append(rel_data)
    
    # 扩展查询：对找到的实体查询其邻居
    for entity_name in list(entity_names)[:5]:  # 只扩展前5个
        try:
            neighbors = get_entity_neighbors(kg, entity_name)
            all_relations.extend(neighbors["outgoing"][:2])
            all_relations.extend(neighbors["incoming"][:2])
        except:
            pass
    
    # 去重关系
    seen_relations = set()
    unique_relations = []
    for rel in all_relations:
        if isinstance(rel, dict):
            rel_key = (rel.get("head", ""), rel.get("relation", ""), rel.get("tail", ""))
            if rel_key and rel_key not in seen_relations:
                seen_relations.add(rel_key)
                unique_relations.append(rel)
    
    return {
        "entities": all_entities[:max_entities],
        "relations": unique_relations[:max_relations],
        "entity_names": list(entity_names),
        "kg_context": format_kg_context_for_rag(all_entities[:max_entities], unique_relations[:max_relations])
    }


def format_kg_context_for_rag(entities: List[Dict], relations: List[Dict]) -> str:
    """
    将知识图谱信息格式化为RAG检索用的上下文
    
    Args:
        entities: 实体列表
        relations: 关系列表
        
    Returns:
        格式化的上下文字符串
    """
    context_parts = []
    
    if entities:
        context_parts.append("知识图谱中的相关概念：")
        for entity in entities:
            name = entity.get("name", "未知")
            data = entity.get("data", {})
            entity_type = data.get("type", "未知类型")
            context_parts.append(f"- {name}（{entity_type}）")
        context_parts.append("")
    
    if relations:
        context_parts.append("相关关系：")
        for rel in relations[:10]:  # 只显示前10个关系
            head = rel.get("head", "未知")
            relation = rel.get("relation", "未知关系")
            tail = rel.get("tail", "未知")
            context_parts.append(f"- {head} {relation} {tail}")
        context_parts.append("")
    
    return "\n".join(context_parts)


def retrieve_docs_with_kg_guidance(vectorstore, kg_info: Dict, question: str, k: int = 5, relevance_threshold: float = 0.6) -> Tuple[List[Document], bool]:
    """
    基于知识图谱信息指导RAG检索教材内容，并检查相关性
    
    Args:
        vectorstore: 向量库对象
        kg_info: 知识图谱定位的信息
        question: 原始问题
        k: 检索文档数量
        relevance_threshold: 相关性阈值（L2距离，越小越相似）
        
    Returns:
        (检索到的文档列表, 是否相关)
    """
    # 构建检索查询：结合原始问题和知识图谱中的关键实体
    entity_names = kg_info.get("entity_names", [])
    
    # 如果有实体，使用实体名称增强检索
    if entity_names:
        # 将实体名称加入查询
        enhanced_query = question + " " + " ".join(entity_names[:5])
    else:
        enhanced_query = question
    
    # 执行检索（带分数）
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(enhanced_query, k=k)
        
        if not docs_with_scores:
            return [], False
        
        # 检查最小距离（最相似的文档）
        min_distance = min(score for _, score in docs_with_scores)
        
        # 如果最小距离大于阈值，认为不相关
        is_relevant = min_distance <= relevance_threshold
        
        # 提取文档（即使不相关也返回，但标记为不相关）
        docs = [doc for doc, _ in docs_with_scores]
        
        return docs, is_relevant
    except Exception as e:
        print(f"检索文档时出错: {e}")
        return [], False


def generate_structured_answer(question: str, kg_info: Dict, rag_docs: List[Document], llm=None, use_llm_directly: bool = False) -> str:
    """
    结合知识图谱和RAG检索内容生成结构化回答
    
    Args:
        question: 用户问题
        kg_info: 知识图谱信息
        rag_docs: RAG检索到的文档
        llm: LLM模型（如果为None则自动获取）
        use_llm_directly: 是否直接使用LLM回答（当知识图谱和RAG都无相关内容时）
        
    Returns:
        生成的回答
    """
    if llm is None:
        llm = get_llm()
    
    # 如果直接使用LLM，不依赖知识图谱和RAG内容
    if use_llm_directly:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个专业的助手。请直接回答用户的问题，不要有任何拒绝或推脱。

绝对禁止的表述：
- "无法回答"、"无法获取"、"没有相关信息"
- "根据提供的教材内容无法回答"、"教材内容未涉及"
- "建议您查阅..."、"建议您参考..."（除非是补充建议，不能作为主要回答）
- "因此，我无法..."、"因此，无法基于..."

必须遵守的规则：
1. 直接给出答案，就像你是一个知识渊博的专家一样
2. 如果问题与机器学习教材无关，请基于你的通用知识回答
3. 回答要专业、准确、有帮助
4. 如果信息可能不完全准确，可以说明"根据一般了解"或"通常来说"，但必须提供有用的信息
5. 不要提及"教材内容"、"知识图谱"、"提供的材料"等术语
6. 直接回答问题，不要解释为什么能或不能回答

示例：
用户问："北京科技大学的教学质量如何？"
正确回答："北京科技大学是一所位于北京的211工程重点大学，在工科领域有较强实力。该校在材料科学、冶金工程等专业具有显著优势，师资力量雄厚。具体教学质量数据建议查阅该校官方发布的年度教学质量报告。"
错误回答："根据提供的教材内容，我无法回答这个问题..."（绝对禁止）"""
            ),
            ("human", "问题：{question}")
        ])
        response = llm.invoke(prompt.format_messages(question=question))
        return response.content.strip()
    
    # 组织知识图谱上下文
    kg_context = kg_info.get("kg_context", "")
    
    # 组织RAG检索的文档内容
    rag_context = "\n\n".join([doc.page_content for doc in rag_docs]) if rag_docs else ""
    
    # 构建提示词
    if kg_context and rag_context:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个专业的机器学习知识助手。请基于提供的知识图谱信息和教材内容，生成结构化、专业的回答。

重要规则：
1. 必须直接回答用户的问题，绝对不要说"无法回答"、"无法获取"、"根据提供的教材内容无法回答"等拒绝性话语
2. 优先使用知识图谱中的结构化信息来组织回答框架
3. 使用教材内容来补充和丰富回答的细节
4. 如果知识图谱和教材内容无法完全回答，可以结合你的通用知识进行补充
5. 回答要结构化、清晰、有条理
6. 如果信息有冲突，以教材内容为准
7. 不要提及"知识图谱"、"教材内容"等术语，直接给出答案

知识图谱信息（结构化概念和关系）：
{kg_context}

教材内容（详细说明）：
{rag_context}"""
            ),
            ("human", "问题：{question}")
        ])
        
        response = llm.invoke(prompt.format_messages(
            question=question,
            kg_context=kg_context,
            rag_context=rag_context
        ))
    elif kg_context:
        # 只有知识图谱信息
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个专业的机器学习知识助手。请基于提供的知识图谱信息回答问题。

重要规则：
1. 必须直接回答用户的问题，绝对不要说"无法回答"、"无法获取"等拒绝性话语
2. 如果知识图谱信息无法完全回答，可以结合你的通用知识进行补充
3. 不要提及"知识图谱"等术语，直接给出答案

知识图谱信息：
{kg_context}"""
            ),
            ("human", "问题：{question}")
        ])
        response = llm.invoke(prompt.format_messages(question=question, kg_context=kg_context))
    elif rag_context:
        # 只有RAG检索内容
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个专业的机器学习知识助手。请基于提供的教材内容回答问题。

重要规则：
1. 必须直接回答用户的问题，绝对不要说"无法回答"、"无法获取"、"根据提供的教材内容无法回答"等拒绝性话语
2. 如果教材内容无法完全回答，可以结合你的通用知识进行补充
3. 不要提及"教材内容"等术语，直接给出答案

教材内容：
{rag_context}"""
            ),
            ("human", "问题：{question}")
        ])
        response = llm.invoke(prompt.format_messages(question=question, rag_context=rag_context))
    else:
        # 都没有，使用通用回答（这种情况理论上不应该发生，因为会在hybrid_kg_rag_answer中判断）
        # 但为了安全起见，仍然提供直接LLM回答
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个专业的助手。请直接回答用户的问题，不要有任何拒绝或推脱。

绝对禁止的表述：
- "无法回答"、"无法获取"、"没有相关信息"
- "根据提供的教材内容无法回答"、"教材内容未涉及"
- "建议您查阅..."（除非是补充建议，不能作为主要回答）
- "因此，我无法..."、"因此，无法基于..."

必须遵守的规则：
1. 直接给出答案，就像你是一个知识渊博的专家一样
2. 回答要专业、准确、有帮助
3. 如果信息可能不完全准确，可以说明"根据一般了解"，但必须提供有用的信息
4. 直接回答问题，不要解释为什么能或不能回答"""
            ),
            ("human", "问题：{question}")
        ])
        response = llm.invoke(prompt.format_messages(question=question))
    
    return response.content.strip()


def check_kg_relevance(kg_info: Dict, min_entities: int = 1, min_relations: int = 0) -> bool:
    """
    检查知识图谱信息是否相关
    
    Args:
        kg_info: 知识图谱信息
        min_entities: 最小实体数量阈值
        min_relations: 最小关系数量阈值
        
    Returns:
        是否相关
    """
    entities = kg_info.get("entities", [])
    relations = kg_info.get("relations", [])
    
    # 如果实体数量或关系数量达到阈值，认为相关
    return len(entities) >= min_entities or len(relations) >= min_relations


def hybrid_kg_rag_answer(question: str, kg: KnowledgeGraph, vectorstore, llm=None) -> str:
    """
    混合架构：知识图谱 + RAG 生成回答
    
    流程：
    1. LLM理解问题
    2. 知识图谱定位相关知识
    3. RAG检索教材内容
    4. 检查相关性，如果都不相关则直接使用LLM
    5. LLM生成结构化回答
    
    Args:
        question: 用户问题
        kg: 知识图谱对象
        vectorstore: 向量库对象
        llm: LLM模型（如果为None则自动获取）
        
    Returns:
        生成的回答
    """
    if llm is None:
        llm = get_llm()
    
    # 步骤1: LLM理解问题
    print("步骤1: LLM理解问题...")
    question_analysis = understand_question(question, llm)
    key_concepts = question_analysis.get("key_concepts", [question])
    print(f"  提取的关键概念: {key_concepts}")
    
    # 步骤2: 知识图谱定位相关知识
    print("步骤2: 知识图谱定位相关知识...")
    kg_info = locate_knowledge_in_kg(kg, key_concepts)
    kg_entities_count = len(kg_info['entities'])
    kg_relations_count = len(kg_info['relations'])
    print(f"  找到 {kg_entities_count} 个实体, {kg_relations_count} 个关系")
    
    # 步骤3: RAG检索教材内容
    print("步骤3: RAG检索教材内容...")
    rag_docs, rag_relevant = retrieve_docs_with_kg_guidance(vectorstore, kg_info, question, k=5)
    print(f"  检索到 {len(rag_docs)} 个文档片段，相关性: {'相关' if rag_relevant else '不相关'}")
    
    # 步骤4: 检查相关性
    kg_relevant = check_kg_relevance(kg_info, min_entities=1, min_relations=0)
    
    print(f"  相关性判断: 知识图谱={kg_relevant}, RAG={rag_relevant}")
    
    # 判断逻辑：
    # - 如果知识图谱和RAG其中一个不相关 → 直接调用LLM回答
    # - 如果两个都相关 → 使用知识图谱和/或RAG内容生成回答
    if not (kg_relevant and rag_relevant):
        # 如果其中一个不相关，直接使用LLM回答
        print("步骤4: 知识图谱和RAG其中一个不相关，直接使用LLM回答...")
        answer = generate_structured_answer(question, kg_info, rag_docs, llm, use_llm_directly=True)
        return answer
    
    # 步骤5: 两个都相关，使用知识图谱和RAG内容生成回答
    print("步骤4: 知识图谱和RAG都相关，使用知识图谱和RAG内容生成回答...")
    answer = generate_structured_answer(question, kg_info, rag_docs, llm, use_llm_directly=False)
    
    return answer


def build_kg_rag_chain(kg: KnowledgeGraph = None, kg_type: str = "ml_full"):
    """
    构建基于知识图谱的RAG链
    
    Args:
        kg: 知识图谱对象（如果为None则自动加载）
        kg_type: 知识图谱类型，"general" 或 "ml_full"
        
    Returns:
        RAG链函数
    """
    import os
    import tempfile
    from pathlib import Path
    
    # 如果没有提供知识图谱，尝试加载
    if kg is None:
        base_dir = Path(__file__).resolve().parent.parent
        env_cache = os.environ.get("VECTOR_CACHE_DIR")
        if env_cache:
            cache_dir = Path(env_cache).parent / "kg_cache"
        else:
            cache_dir = Path(tempfile.gettempdir()) / "kg_cache"
        
        if kg_type == "ml_full":
            kg_path = cache_dir / "ml_full_kg"
        else:
            kg_path = cache_dir / "knowledge_graph"
        
        kg = KnowledgeGraph()
        try:
            kg.load(kg_path)
            print(f"✓ 已加载知识图谱：{kg_path}")
        except Exception as e:
            print(f"✗ 加载知识图谱失败: {e}")
            print("提示：请先运行 python src/build_ml_kg.py 构建知识图谱")
            return None
    
    llm = get_llm()
    
    def kg_rag_chain(inputs: Dict) -> str:
        """基于知识图谱的RAG链"""
        question = inputs.get("question", "")
        if not question:
            return "请提供一个问题。"
        
        return generate_answer_from_kg(kg, question, llm)
    
    return kg_rag_chain

