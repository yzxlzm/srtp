# 混合架构（知识图谱 + RAG）使用指南

## 架构概述

本系统采用混合架构，结合知识图谱和RAG的优势，提供更准确、更结构化的回答。

### 工作流程

```
用户提问
    ↓
LLM理解问题（提取关键概念和查询意图）
    ↓
知识图谱定位相关知识（查找相关实体和关系）
    ↓
RAG检索教材内容（基于知识图谱指导的检索）
    ↓
LLM生成结构化回答（结合知识图谱和RAG内容）
```

## 核心优势

1. **精准定位**：通过知识图谱快速定位相关概念和关系
2. **内容补充**：通过RAG检索获取详细的教材内容
3. **结构化回答**：结合两者生成结构清晰、信息丰富的回答
4. **智能检索**：使用知识图谱中的实体名称增强RAG检索效果

## 使用方法

### 方法一：通过API接口（推荐）

```bash
# 启动服务
python src/main.py

# 调用查询接口
curl -X POST "http://127.0.0.1:8000/query?kg_type=ml_chapters" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是监督学习？监督学习和无监督学习有什么区别？"}'
```

### 方法二：在代码中使用

```python
from knowledge_graph import KnowledgeGraph, hybrid_kg_rag_answer
from rag_pipeline import build_rag_chain
from vectorstore import load_vectorstore
from embeddings import get_embeddings
import os
import tempfile
from pathlib import Path

# 加载知识图谱
kg = KnowledgeGraph()
kg.load(Path(tempfile.gettempdir()) / "kg_cache" / "ml_chapters_2_kg")

# 加载向量库
vectorstore = load_vectorstore(
    Path(tempfile.gettempdir()) / "vectorstore_cache",
    get_embeddings()
)

# 使用混合架构生成回答
question = "什么是监督学习？"
answer = hybrid_kg_rag_answer(question, kg, vectorstore)
print(answer)
```

## 核心函数说明

### `understand_question(question, llm=None)`

使用LLM理解问题，提取关键概念和查询意图。

**参数：**
- `question`: 用户问题
- `llm`: LLM模型（可选）

**返回：**
```python
{
    "key_concepts": ["概念1", "概念2"],  # 关键概念列表
    "query_type": "类型",  # 问题类型
    "search_keywords": ["关键词1", "关键词2"],  # 搜索关键词
    "intent": "查询意图描述"
}
```

### `locate_knowledge_in_kg(kg, key_concepts, max_entities=15, max_relations=20)`

在知识图谱中定位相关知识。

**参数：**
- `kg`: 知识图谱对象
- `key_concepts`: 关键概念列表
- `max_entities`: 最大实体数量
- `max_relations`: 最大关系数量

**返回：**
```python
{
    "entities": [...],  # 相关实体列表
    "relations": [...],  # 相关关系列表
    "entity_names": [...],  # 实体名称列表（用于RAG检索）
    "kg_context": "格式化的上下文"  # 用于RAG的上下文
}
```

### `retrieve_docs_with_kg_guidance(vectorstore, kg_info, question, k=5)`

基于知识图谱信息指导RAG检索教材内容。

**参数：**
- `vectorstore`: 向量库对象
- `kg_info`: 知识图谱定位的信息
- `question`: 原始问题
- `k`: 检索文档数量

**返回：**
- 检索到的文档列表

### `generate_structured_answer(question, kg_info, rag_docs, llm=None)`

结合知识图谱和RAG检索内容生成结构化回答。

**参数：**
- `question`: 用户问题
- `kg_info`: 知识图谱信息
- `rag_docs`: RAG检索到的文档
- `llm`: LLM模型（可选）

**返回：**
- 生成的回答字符串

### `hybrid_kg_rag_answer(question, kg, vectorstore, llm=None)`

混合架构主函数，执行完整的流程。

**参数：**
- `question`: 用户问题
- `kg`: 知识图谱对象
- `vectorstore`: 向量库对象
- `llm`: LLM模型（可选）

**返回：**
- 生成的回答字符串

## 工作流程详解

### 步骤1：LLM理解问题

系统首先使用LLM分析用户问题，提取：
- **关键概念**：问题中涉及的核心概念
- **查询类型**：问题的类型（定义、比较、应用等）
- **搜索关键词**：用于后续检索的关键词
- **查询意图**：问题的意图描述

**示例：**
```
问题："什么是监督学习？监督学习和无监督学习有什么区别？"

理解结果：
{
    "key_concepts": ["监督学习", "无监督学习"],
    "query_type": "定义和比较",
    "search_keywords": ["监督学习", "无监督学习", "区别"],
    "intent": "了解监督学习的定义，并比较监督学习和无监督学习"
}
```

### 步骤2：知识图谱定位相关知识

基于提取的关键概念，在知识图谱中查找：
- **相关实体**：与关键概念相关的实体
- **相关关系**：实体之间的关系
- **实体邻居**：相关实体的邻居节点（扩展查询）

**示例：**
```
找到实体：
- 监督学习（概念/术语）
- 无监督学习（概念/术语）
- 分类（问题/任务）
- 回归（问题/任务）

找到关系：
- 监督学习 --[用于]--> 分类
- 监督学习 --[用于]--> 回归
- 监督学习 --[需要]--> 标注数据
```

### 步骤3：RAG检索教材内容

使用知识图谱中的实体名称增强检索查询，从向量库中检索相关文档片段。

**检索策略：**
- 原始问题 + 知识图谱中的实体名称
- 提高检索的准确性和相关性

**示例：**
```
增强查询："什么是监督学习？监督学习和无监督学习有什么区别？ 监督学习 无监督学习 分类 回归"

检索结果：5个相关的文档片段
```

### 步骤4：LLM生成结构化回答

结合知识图谱的结构化信息和RAG检索的详细内容，生成结构化回答。

**生成策略：**
1. 使用知识图谱信息构建回答框架
2. 使用RAG内容补充细节
3. 确保回答结构清晰、信息完整

## 与单一架构的对比

| 特性 | 纯知识图谱 | 纯RAG | 混合架构 |
|------|-----------|-------|---------|
| 回答准确性 | 高（结构化） | 中（依赖检索） | 高（结合两者） |
| 内容详细度 | 中（结构化信息） | 高（完整文档） | 高（结构化+详细） |
| 检索效率 | 高（图查询） | 中（向量检索） | 高（图指导检索） |
| 可解释性 | 高（关系路径） | 中（文档片段） | 高（两者结合） |

## 配置说明

### 知识图谱配置

默认使用 `ml_chapters` 类型的知识图谱（机器学习前两章）。

可以通过 `kg_type` 参数选择：
- `ml_chapters`: 机器学习前两章知识图谱
- `general`: 通用知识图谱

### 向量库配置

向量库会自动加载，如果不存在会自动构建。

缓存位置：
- Windows: `C:\Users\<用户名>\AppData\Local\Temp\vectorstore_cache\`
- Linux/Mac: `/tmp/vectorstore_cache/`

## 性能优化建议

1. **预加载**：系统启动时会自动加载知识图谱和向量库
2. **缓存机制**：知识图谱和向量库都支持缓存，避免重复构建
3. **参数调整**：根据问题复杂度调整 `max_entities` 和 `max_relations` 参数

## 故障排查

### 问题：知识图谱未找到

**解决方案：**
```bash
python src/build_ml_kg.py
```

### 问题：向量库未找到

**解决方案：**
- 系统会自动构建向量库
- 确保 `data/` 目录下有文档文件

### 问题：回答质量不佳

**可能原因：**
- 知识图谱信息不完整
- RAG检索结果不相关

**解决方案：**
- 检查知识图谱构建质量
- 调整检索参数（k值）
- 检查文档内容是否完整

## 示例问题

以下问题适合使用混合架构：

- "什么是监督学习？"
- "监督学习和无监督学习有什么区别？"
- "什么是过拟合？如何解决过拟合问题？"
- "请介绍一下决策树算法，它有什么优缺点？"
- "什么是交叉验证？它有什么作用？"
- "准确率和精确率有什么区别？"

## 相关文件

- `src/knowledge_graph.py` - 知识图谱核心模块（包含混合架构函数）
- `src/api.py` - API接口定义
- `src/rag_pipeline.py` - RAG管道
- `src/vectorstore.py` - 向量库管理

