from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from neo4j import GraphDatabase

from app_config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = NEO4J_URI
    username: str = NEO4J_USERNAME
    password: Optional[str] = NEO4J_PASSWORD
    database: str = NEO4J_DATABASE


class Neo4jStore:
    """
    轻量 Neo4j 存储层：用于将现有 KnowledgeGraph（entities/relations）落库，并支持按关键词检索子图。
    """

    def __init__(self, cfg: Optional[Neo4jConfig] = None):
        self.cfg = cfg or Neo4jConfig()
        if not self.cfg.password:
            raise ValueError("NEO4J_PASSWORD 未设置，请在 .env 中配置")
        self._driver = GraphDatabase.driver(self.cfg.uri, auth=(self.cfg.username, self.cfg.password))

    def close(self):
        self._driver.close()

    def ensure_schema(self):
        """
        创建必要的约束与索引。
        - Entity(name) 唯一
        - 便于 name 模糊查询的索引
        """
        queries = [
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        with self._driver.session(database=self.cfg.database) as session:
            for q in queries:
                session.run(q)

    def upsert_entities(self, entities: Dict[str, Dict]):
        """
        entities: KnowledgeGraph.entities 的结构：{name: {type, properties}}
        """
        rows = []
        for name, data in entities.items():
            rows.append(
                {
                    "name": name,
                    "type": data.get("type", "Unknown"),
                    "properties": data.get("properties", {}) or {},
                }
            )

        cypher = """
        UNWIND $rows AS row
        MERGE (e:Entity {name: row.name})
        SET e.type = row.type
        SET e += row.properties
        """
        with self._driver.session(database=self.cfg.database) as session:
            session.run(cypher, rows=rows)

    def upsert_relations(self, relations: List[Dict]):
        """
        relations: KnowledgeGraph.relations 的结构：[{head, relation, tail, source}]
        说明：关系类型在 Neo4j 里需要是合法 label/name，这里统一用 :RELATED_TO，并把原 relation 放在属性 rel_type。
        """
        rows = []
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            if not head or not tail:
                continue
            rows.append(
                {
                    "head": head,
                    "tail": tail,
                    "rel_type": rel.get("relation", "Unknown"),
                    "source": rel.get("source"),
                }
            )

        cypher = """
        UNWIND $rows AS row
        MATCH (h:Entity {name: row.head})
        MATCH (t:Entity {name: row.tail})
        MERGE (h)-[r:RELATED_TO {rel_type: row.rel_type}]->(t)
        SET r.source = row.source
        """
        with self._driver.session(database=self.cfg.database) as session:
            session.run(cypher, rows=rows)

    def clear(self):
        with self._driver.session(database=self.cfg.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def stats(self) -> Dict:
        with self._driver.session(database=self.cfg.database) as session:
            node_cnt = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rel_cnt = session.run("MATCH (:Entity)-[r:RELATED_TO]->(:Entity) RETURN count(r) AS c").single()["c"]
        return {"entities": node_cnt, "relations": rel_cnt}

    def search_entities(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        简单的子串匹配搜索（可后续升级为全文索引/向量检索）。
        """
        kw = keyword.strip()
        if not kw:
            return []
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($kw)
        RETURN e.name AS name, e.type AS type
        LIMIT $limit
        """
        with self._driver.session(database=self.cfg.database) as session:
            res = session.run(cypher, kw=kw, limit=limit)
            return [dict(r) for r in res]

    def fetch_subgraph_by_entities(self, entity_names: List[str], hops: int = 1, limit_rels: int = 50) -> List[Tuple[str, str, str]]:
        """
        给定实体列表，从 Neo4j 拉取其邻域的三元组 (head, rel_type, tail)。
        """
        names = [n for n in entity_names if n]
        if not names:
            return []

        # 目前只支持 1 hop（更深 hop 可后续用可变长度路径 + 去重）
        if hops != 1:
            hops = 1

        cypher = """
        MATCH (e:Entity)-[r:RELATED_TO]->(t:Entity)
        WHERE e.name IN $names
        RETURN e.name AS head, r.rel_type AS relation, t.name AS tail
        LIMIT $limit_rels
        """
        with self._driver.session(database=self.cfg.database) as session:
            res = session.run(cypher, names=names, limit_rels=limit_rels)
            return [(r["head"], r["relation"], r["tail"]) for r in res]

