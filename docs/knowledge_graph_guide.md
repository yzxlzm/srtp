# 知识图谱构建指南

## 概述

知识图谱用于从文档中提取实体（如学校、专业、企业、职位等）和关系（如就业于、毕业于、招聘等），构建结构化的知识网络。

## 安装依赖

```bash
pip install networkx
```

## 使用方法

### 1. 构建知识图谱

#### 方法一：通过 API（推荐）

```bash
# 启动应用
python src/main.py

# 构建知识图谱
curl -X POST http://127.0.0.1:8000/kg/build
```

#### 方法二：在代码中使用

```python
from knowledge_graph import build_knowledge_graph_from_docs

# 从 data 目录的文档构建知识图谱
kg = build_knowledge_graph_from_docs()
```

### 2. 查询知识图谱

#### 获取统计信息

```bash
curl http://127.0.0.1:8000/kg/stats
```

返回示例：
```json
{
  "status": "success",
  "statistics": {
    "total_entities": 150,
    "total_relations": 200,
    "entity_types": {
      "学校": 10,
      "专业": 50,
      "企业": 30,
      "职位": 60
    },
    "relation_types": {
      "就业于": 80,
      "毕业于": 50,
      "招聘": 70
    }
  }
}
```

#### 查询实体和关系

```bash
curl -X POST http://127.0.0.1:8000/kg/query \
  -H "Content-Type: application/json" \
  -d '{"question": "北京科技大学"}'
```

### 3. 知识图谱存储位置

知识图谱默认保存在：
- Windows: `C:\Users\<用户名>\AppData\Local\Temp\kg_cache\knowledge_graph\`
- Linux/Mac: `/tmp/kg_cache/knowledge_graph/`

或通过环境变量自定义：
```bash
export VECTOR_CACHE_DIR=/path/to/cache
# 知识图谱会保存在 /path/to/cache/../kg_cache/
```

## 实体和关系类型

### 实体类型
- **学校/高校**：如"北京科技大学"
- **专业/学科**：如"计算机科学"
- **企业/公司**：如"腾讯"
- **职位/岗位**：如"软件工程师"
- **毕业生/学生**：如"2023届毕业生"
- **地区/城市**：如"北京"
- **行业**：如"互联网"
- **技能/能力**：如"Python编程"

### 关系类型
- **就业于/就职于**：毕业生 → 企业
- **毕业于**：毕业生 → 学校
- **招聘**：企业 → 职位
- **位于**：学校/企业 → 地区
- **属于**：专业 → 学校
- **需要**：职位 → 技能
- **掌握**：毕业生 → 技能
- **从事**：毕业生 → 职位

## 工作流程

1. **文档加载**：从 `data/` 目录加载所有文档
2. **文本分割**：将文档分割成小块
3. **实体关系提取**：使用 LLM 从每个文本块中提取实体和关系
4. **知识图谱构建**：将提取的实体和关系组织成图结构
5. **持久化存储**：保存为 JSON 和 NetworkX 图格式

## 性能优化

- **批量处理**：默认每批处理 5 个文档，可通过 `batch_size` 参数调整
- **缓存机制**：构建完成后会缓存，下次启动直接加载
- **增量更新**：删除缓存文件后重新构建即可更新知识图谱

## 与 RAG 系统集成

知识图谱可以与现有的 RAG 系统结合使用：

1. **增强检索**：使用知识图谱查找相关实体，然后检索相关文档
2. **关系推理**：通过知识图谱的关系网络进行推理
3. **可视化展示**：展示实体之间的关系网络

## 注意事项

1. **构建时间**：首次构建可能需要较长时间，取决于文档数量和 LLM API 响应速度
2. **API 调用**：构建过程会调用大量 LLM API，注意 API 配额
3. **准确性**：实体和关系的提取依赖于 LLM 的准确性，可能需要人工校验

## 示例代码

```python
from knowledge_graph import KnowledgeGraph, build_knowledge_graph_from_docs, query_knowledge_graph

# 构建知识图谱
kg = build_knowledge_graph_from_docs()

# 查询
results = query_knowledge_graph(kg, "北京科技大学")
for result in results:
    print(result)

# 获取统计信息
stats = kg.get_statistics()
print(f"实体数量: {stats['total_entities']}")
print(f"关系数量: {stats['total_relations']}")
```

