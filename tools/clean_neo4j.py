"""Xoá toàn bộ Node + Relationship trong Neo4j rồi recreate schema.

Dùng CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS —
tránh timeout khi số lượng lớn (không dùng MATCH (n) DETACH DELETE n
trống tay vì với 200k+ node AuraDB sẽ timeout).

Chạy:
    python tools/clean_neo4j.py           # hỏi xác nhận
    python tools/clean_neo4j.py --force   # bỏ qua xác nhận (Colab scripted)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

SCHEMA_INIT = Path(__file__).parent.parent / "load" / "schema_init.cypher"
BATCH = 5000


def _count_nodes(session) -> int:
    return session.run("MATCH (n) RETURN count(n) AS c").single()["c"]


def _count_edges(session) -> int:
    return session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]


def clean_all(force: bool = False) -> None:
    from load.neo4j_client import Neo4jClient

    with Neo4jClient() as client:
        with client._driver.session() as session:
            node_count = _count_nodes(session)
            edge_count = _count_edges(session)

        if node_count == 0 and edge_count == 0:
            print("DB đã trống (0 node, 0 edge) — không cần xoá.")
            _recreate_schema(client)
            return

        print(f"Hiện tại: {node_count:,} node, {edge_count:,} edge.")

        if not force:
            ans = input("Bạn sắp xoá TOÀN BỘ dữ liệu trong Neo4j. Gõ 'yes' để tiếp tục: ")
            if ans.strip().lower() != "yes":
                print("Huỷ.")
                return

        print(f"Bắt đầu xoá theo batch {BATCH} node ...")
        t0 = time.time()
        deleted_total = 0

        while True:
            with client._driver.session() as session:
                result = session.run(
                    f"MATCH (n) WITH n LIMIT {BATCH} CALL {{ WITH n DETACH DELETE n }} IN TRANSACTIONS OF {BATCH} ROWS RETURN count(*) AS deleted"
                )
                deleted = result.single()["deleted"]

            deleted_total += deleted
            if deleted == 0:
                break

            with client._driver.session() as session:
                remaining = _count_nodes(session)
            elapsed = time.time() - t0
            print(f"  Đã xoá {deleted_total:,} node | còn {remaining:,} | {elapsed:.0f}s")

        with client._driver.session() as session:
            final_nodes = _count_nodes(session)
            final_edges = _count_edges(session)

        elapsed = time.time() - t0
        print(f"\nXoá xong: {final_nodes} node, {final_edges} edge còn lại ({elapsed:.1f}s).")

        _recreate_schema(client)


def _recreate_schema(client) -> None:
    print("Recreate schema (constraints + index) ...")
    client.run_schema_init(SCHEMA_INIT)
    print("Schema OK.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Xoá toàn bộ Neo4j rồi recreate schema")
    parser.add_argument("--force", action="store_true", help="Bỏ qua xác nhận (dùng trên Colab)")
    args = parser.parse_args()
    clean_all(force=args.force)


if __name__ == "__main__":
    main()
