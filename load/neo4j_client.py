"""Driver wrapper cho Neo4j AuraDB — batch UNWIND + MERGE, idempotent."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from config import NEO4J_BATCH_SIZE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

logger = logging.getLogger(__name__)

_WRITE_MAX_RETRIES = 3
_WRITE_RETRY_WAIT = 5  # giây chờ trước khi retry

# Exception types cần retry: Neo4j-level + socket-level (TimeoutError, OSError)
_RETRYABLE = (SessionExpired, ServiceUnavailable, TransientError, TimeoutError, OSError)


class Neo4jClient:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=30,        # timeout thiết lập kết nối mới (giây)
            max_connection_lifetime=300,  # đóng connection sau 5 phút tránh server kill âm thầm
            keep_alive=True,
        )

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

    def query(self, cypher: str, **params) -> list[dict]:
        """Read-only query — trả về list[dict] của các record kết quả."""
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def batch_write(self, cypher: str, rows: list[dict], batch_size: int = NEO4J_BATCH_SIZE) -> None:
        """Chạy `UNWIND $rows AS row ...` theo từng batch — pattern dùng chung
        cho mọi hàm load_* trong loaders.py, tránh viết lại boilerplate.

        Mỗi batch dùng session riêng để tránh session chết làm hỏng toàn bộ.
        Retry tối đa _WRITE_MAX_RETRIES lần khi gặp SessionExpired / timeout.
        """
        if not rows:
            return
        total_batches = (len(rows) + batch_size - 1) // batch_size
        for idx, i in enumerate(range(0, len(rows), batch_size)):
            batch = rows[i : i + batch_size]
            for attempt in range(1, _WRITE_MAX_RETRIES + 1):
                try:
                    with self._driver.session() as session:
                        session.execute_write(
                            lambda tx, _b=batch: tx.run(cypher, rows=_b).consume()
                        )
                    break
                except _RETRYABLE as exc:
                    if attempt < _WRITE_MAX_RETRIES:
                        logger.warning(
                            "batch_write lỗi (batch %d/%d, lần %d/%d) — chờ %ds rồi retry: %s",
                            idx + 1, total_batches, attempt, _WRITE_MAX_RETRIES, _WRITE_RETRY_WAIT, exc,
                        )
                        time.sleep(_WRITE_RETRY_WAIT)
                    else:
                        logger.error(
                            "batch_write thất bại sau %d lần retry (batch %d/%d) — bỏ qua batch này.",
                            _WRITE_MAX_RETRIES, idx + 1, total_batches,
                        )
        logger.debug("batch_write: %d dòng, batch_size=%d", len(rows), batch_size)
