# 基于知识图谱的内容生成指南

## 概述

本系统支持基于知识图谱生成内容，能够从构建好的知识图谱中提取相关信息，并使用大语言模型生成专业的回答。

## 功能特点

1. **智能上下文提取**：自动从知识图谱中提取与问题相关的实体和关系
2. **关系扩展**：自动查询相关实体的邻居节点，获取更完整的上下文
3. **专业回答生成**：基于知识图谱信息生成准确、专业的回答
4. **多种使用方式**：支持直接调用、RAG链、API接口等多种使用方式

## 使用方法

### 方法一：直接使用函数（推荐）

```python
from knowledge_graph import KnowledgeGraph, generate_answer_from_kg, load_ml_kg

# 加载知识图谱
kg = load_ml_kg()

# 生成回答
question = "什么是监督学习？"
answer = generate_answer_from_kg(kg, question)
print(answer)
```

### 方法二：使用RAG链

```python
from knowledge_graph import build_kg_rag_chain

# 构建RAG链
chain = build_kg_rag_chain(kg_type="ml_chapters")

# 使用链生成回答
answer = chain({"question": "什么是过拟合？"})
print(answer)
```

### 方法三：通过API接口

#### 1. 启动API服务

```bash
python src/main.py
# 或
uvicorn src.api:app --reload --port 8000
```

#### 2. 调用生成接口

```bash
# 使用curl
curl -X POST "http://127.0.0.1:8000/kg/generate?kg_type=ml_chapters" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是监督学习？"}'

# 使用Python requests
import requests

response = requests.post(
    "http://127.0.0.1:8000/kg/generate",
    params={"kg_type": "ml_chapters"},
    json={"question": "什么是监督学习？"}
)
print(response.json())
```

### 方法四：运行示例脚本

```bash
# 运行完整示例
python src/generate_from_kg_example.py
```

## 核心函数说明

### `generate_answer_from_kg(kg, question, llm=None)`

基于知识图谱生成回答。

**参数：**
- `kg`: 知识图谱对象
- `question`: 用户问题
- `llm`: LLM模型（可选，默认自动获取）

**返回：**
- 生成的回答字符串

**示例：**
```python
answer = generate_answer_from_kg(kg, "什么是决策树？")
```

### `extract_kg_context(kg, question, max_entities=10, max_relations=15)`

从知识图谱中提取与问题相关的上下文信息。

**参数：**
- `kg`: 知识图谱对象
- `question`: 用户问题
- `max_entities`: 最大实体数量
- `max_relations`: 最大关系数量

**返回：**
- 格式化的上下文字符串

**示例：**
```python
context = extract_kg_context(kg, "什么是监督学习？")
print(context)
```

### `build_kg_rag_chain(kg=None, kg_type="ml_chapters")`

构建基于知识图谱的RAG链。

**参数：**
- `kg`: 知识图谱对象（可选，默认自动加载）
- `kg_type`: 知识图谱类型，"general" 或 "ml_chapters"

**返回：**
- RAG链函数

**示例：**
```python
chain = build_kg_rag_chain(kg_type="ml_chapters")
answer = chain({"question": "什么是交叉验证？"})
```

## 工作流程

1. **问题理解**：接收用户问题
2. **知识检索**：从知识图谱中查询相关实体和关系
3. **上下文扩展**：查询相关实体的邻居节点，获取更完整的上下文
4. **上下文组织**：将检索到的信息组织成结构化的上下文
5. **回答生成**：使用LLM基于上下文生成专业回答

## 与普通RAG的区别

| 特性 | 普通RAG | 知识图谱RAG |
|------|---------|-------------|
| 数据源 | 原始文档文本 | 结构化实体和关系 |
| 检索方式 | 向量相似度搜索 | 实体关系图查询 |
| 上下文组织 | 文档片段 | 实体-关系网络 |
| 推理能力 | 有限 | 支持关系推理 |
| 可解释性 | 较低 | 较高（可展示关系路径） |

## 使用场景

1. **概念解释**：解释机器学习中的概念和术语
2. **关系查询**：查询实体之间的关系
3. **知识推理**：基于知识图谱进行推理
4. **结构化问答**：回答需要结构化信息的问题

## 示例问题

以下是一些适合使用知识图谱生成回答的问题：

- "什么是监督学习？"
- "监督学习和无监督学习有什么区别？"
- "什么是过拟合？如何解决过拟合问题？"
- "请介绍一下决策树算法"
- "什么是交叉验证？它有什么作用？"
- "准确率和精确率有什么区别？"

## 注意事项

1. **知识图谱构建**：使用前需要先构建知识图谱（运行 `python src/build_ml_kg.py`）
2. **API调用**：生成回答会调用LLM API，注意API配额
3. **上下文限制**：默认最多提取10个实体和15个关系，可通过参数调整
4. **回答质量**：回答质量取决于知识图谱的完整性和准确性

## 性能优化建议

1. **批量处理**：对于多个问题，可以批量处理以提高效率
2. **缓存机制**：知识图谱已支持缓存，避免重复构建
3. **上下文限制**：根据问题复杂度调整 `max_entities` 和 `max_relations` 参数

## 故障排查

### 问题：知识图谱未找到

**解决方案：**
```bash
# 先构建知识图谱
python src/build_ml_kg.py
```

### 问题：生成回答为空

**可能原因：**
- 知识图谱中没有相关问题相关的实体和关系
- LLM API调用失败

**解决方案：**
- 检查知识图谱是否包含相关内容
- 检查API配置和网络连接

### 问题：回答不准确

**可能原因：**
- 知识图谱中的信息不完整
- 上下文提取不充分

**解决方案：**
- 增加 `max_entities` 和 `max_relations` 参数
- 检查知识图谱构建质量

## 相关文件

- `src/knowledge_graph.py` - 知识图谱核心模块
- `src/generate_from_kg_example.py` - 使用示例
- `src/build_ml_kg.py` - 知识图谱构建脚本
- `src/api.py` - API接口定义

