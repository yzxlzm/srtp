import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # 若未安装 python-dotenv，不影响运行
    pass

env = os.getenv

# DeepSeek 基础配置
DEEPSEEK_API_KEY = env("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = env("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# OpenAI 基础配置（需自行在 .env 中设置 OPENAI_API_KEY）
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = env("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Neo4j（GraphRAG）
# 示例：
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=please_change_me
# NEO4J_DATABASE=neo4j
NEO4J_URI = env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = env("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = env("NEO4J_PASSWORD")
NEO4J_DATABASE = env("NEO4J_DATABASE", "neo4j")