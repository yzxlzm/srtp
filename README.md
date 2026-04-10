# LangChain + DeepSeek 示例项目

## 快速开始

1. 复制 `.env.example` 为 `.env` 并填写 `DEEPSEEK_API_KEY`，如需自定义可修改 `DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 与 `BGE_MODEL`。
2. 创建虚拟环境并安装依赖：
```bash
python -m venv .venv
.venv/Scripts/activate       # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```
3. 启动 Web 版问答（首次会构建向量库）：
```bash
python src/main.py
# 浏览器访问 http://127.0.0.1:8000 打开页面
```
4. 仅启动接口（可选）：
```bash
uvicorn src.api:app --reload --port 8000
# POST /query {"question":"什么是 LangChain?"}
```

## 关于 DeepSeek

本项目使用 LangChain 的 `ChatOpenAI` 封装，通过 `openai_api_base` 指向 DeepSeek API，从而兼容 OpenAI 协议的国产模型。

## Neo4j + GraphRAG（更完整知识图谱）

本项目现已支持将本地构建的 `KnowledgeGraph`（实体/关系）同步到 **Neo4j**，并提供一个 **GraphRAG（简化版）** 的查询接口：
- **/neo4j/sync**：把当前知识图谱同步到 Neo4j
- **/graphrag/query**：LLM 抽取关键概念 → Neo4j 拉取子图三元组 → LLM 基于子图回答

### 1) 启动 Neo4j（推荐用 docker-compose）

在项目根目录执行：

```bash
docker compose up -d neo4j
```

- Neo4j Browser：`http://localhost:7474`
- Bolt：`bolt://localhost:7687`

### 2) 配置 .env（示例）

请在 `.env` 中增加：

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=please_change_me
NEO4J_DATABASE=neo4j
```

> 若你用 docker-compose 直接启动 neo4j，默认账号是 `neo4j`，密码由 `NEO4J_PASSWORD` 决定。

### 3) 构建全量机器学习知识图谱并同步

- 构建（全量）：`python src/build_ml_kg.py`  
  或调用接口：`POST /kg/build?kg_type=ml_full`

启动 API 后，同步到 Neo4j：

```bash
curl -X POST "http://127.0.0.1:8000/neo4j/sync?kg_type=ml_chapters"
```

> 如需同步全量图谱，使用 `kg_type=ml_full`

### 4) GraphRAG 查询

```bash
curl -X POST "http://127.0.0.1:8000/graphrag/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"什么是过拟合？如何用正则化缓解？\",\"model\":\"openai-gpt-4o\"}"
```

返回中会包含命中的实体、子图三元组数量，以及最终回答。

## 知识库文档

系统支持以下格式的文档作为知识库：
- `.txt` - 纯文本文件
- `.md` - Markdown 文件
- `.docx` - Word 文档

**使用方法：**
1. 将文档放入 `data/` 目录
2. 启动应用时，系统会自动加载 `data/` 目录下所有支持的文档
3. 首次运行或文档更新后，向量库会自动重建

**示例：**
```bash
# 将 Word 文档放入 data 目录
cp your_document.docx data/

# 重启应用，系统会自动加载并构建向量库
python src/main.py
```

## 后续建议

- 有 GPU 时可将 `HuggingFaceEmbeddings` 的 device 改为 `cuda`。
- 生产部署可换用 Milvus/Chroma 等向量库。
- 若需离线推理，可用 transformers/vLLM 加载本地模型（需大算力）。

