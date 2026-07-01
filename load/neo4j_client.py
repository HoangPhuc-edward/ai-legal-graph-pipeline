"""Driver wrapper cho Neo4j AuraDB — batch UNWIND + MERGE, idempotent."""
from __future__ import annotations

import logging
from pathlib import Path

from neo4j import GraphDatabase

from config import NEO4J_BATCH_SIZE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run_schema_init(self, cypher_path: Path) -> None:
        """Chạy 1 lần khi khởi tạo DB — tạo constraints + index."""
        statements = [
            s.strip()
            for s in cypher_path.read_text(encoding="utf-8").split(";")
            if s.strip()
        ]
        with self._driver.session() as session:
            for statement in statements:
                session.run(statement)
        logger.info("Đã chạy schema_init.cypher (%d statement)", len(statements))

    def count_nodes(self) -> int:
        with self._driver.session() as session:
            return session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

    def count_edges(self) -> int:
        with self._driver.session() as session:
            return session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    def batch_write(self, cypher: str, rows: list[dict], batch_size: int = NEO4J_BATCH_SIZE) -> None:
        """Chạy `UNWIND $rows AS row ...` theo từng batch — pattern dùng chung
        cho mọi hàm load_* trong loaders.py, tránh viết lại boilerplate."""
        if not rows:
            return
        with self._driver.session() as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                session.run(cypher, rows=batch)
        logger.info("Đã ghi %d dòng (batch_size=%d)", len(rows), batch_size)
