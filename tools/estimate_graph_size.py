"""Ước tính số Node / Edge trước khi load lên Neo4j AuraDB.

Đọc file trung gian đã transform (jsonl trong TRANSFORMED_DIR) THEO TỪNG DÒNG —
không deserialise toàn bộ object, không giữ data trong RAM, chỉ đếm và đọc field
"type" nhỏ khi cần phân loại TextUnit. Không kết nối Neo4j, không cần .env.

Chạy:
    python tools/estimate_graph_size.py
    python tools/estimate_graph_size.py --dir path/to/transformed   # custom dir
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRANSFORMED_DIR

NODE_LIMIT = 200_000
EDGE_LIMIT = 400_000


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _count_textunit_types(path: Path) -> tuple[int, int]:
    """Trả về (noi_dung_count, cache_action_count) — đọc từng dòng JSONL, chỉ
    parse field "type" (không load toàn object)."""
    if not path.exists():
        return 0, 0
    noi_dung = cache_action = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                typ = json.loads(line).get("type", "noi_dung")
            except json.JSONDecodeError:
                continue
            if typ == "cache_action":
                cache_action += 1
            else:
                noi_dung += 1
    return noi_dung, cache_action


def estimate(transformed_dir: Path = TRANSFORMED_DIR) -> dict:
    norms = _count_lines(transformed_dir / "norms.jsonl")
    components = _count_lines(transformed_dir / "components.jsonl")
    tu_noi_dung, tu_cache = _count_textunit_types(transformed_dir / "textunits.jsonl")
    actions = _count_lines(transformed_dir / "actions.jsonl")
    action_links = _count_lines(transformed_dir / "action_links.jsonl")
    relations = _count_lines(transformed_dir / "relations.jsonl")

    total_nodes = norms + components + (tu_noi_dung + tu_cache) + actions
    contains_edges = components                    # mỗi Component có đúng 1 cha (Norm hoặc Component)
    has_textunit_comp = tu_noi_dung               # Component -> TextUnit (noi_dung)
    has_textunit_cache = tu_cache                  # Action -> TextUnit (cache_action)
    has_action = action_links                      # Component A -> Action
    apply_to = action_links                        # Action -> Component B
    total_edges = contains_edges + has_textunit_comp + has_textunit_cache + has_action + apply_to + relations

    return {
        "nodes": {
            "Norm": norms,
            "Component": components,
            "TextUnit (noi_dung)": tu_noi_dung,
            "TextUnit (cache_action)": tu_cache,
            "Action": actions,
            "TOTAL": total_nodes,
        },
        "edges": {
            "CONTAINS": contains_edges,
            "HAS_TEXTUNIT (Component)": has_textunit_comp,
            "HAS_TEXTUNIT (Action cache)": has_textunit_cache,
            "HAS_ACTION": has_action,
            "APPLY_TO": apply_to,
            "NormRelation": relations,
            "TOTAL": total_edges,
        },
    }


def _fmt(n: int) -> str:
    return f"{n:>10,}"


def print_report(stats: dict) -> None:
    print("\n=== Node ===")
    for label, count in stats["nodes"].items():
        marker = "──────────" if label == "TOTAL" else ""
        print(f"  {label:<30} {_fmt(count)} {marker}")
    print("\n=== Edge ===")
    for label, count in stats["edges"].items():
        marker = "──────────" if label == "TOTAL" else ""
        print(f"  {label:<30} {_fmt(count)} {marker}")
    print()

    node_total = stats["nodes"]["TOTAL"]
    edge_total = stats["edges"]["TOTAL"]
    node_ok = node_total <= NODE_LIMIT
    edge_ok = edge_total <= EDGE_LIMIT
    pct_node = node_total / NODE_LIMIT * 100
    pct_edge = edge_total / EDGE_LIMIT * 100

    status_n = "OK " if node_ok else "!!!"
    status_e = "OK " if edge_ok else "!!!"
    print(f"[{status_n}] Nodes  {node_total:>10,} / {NODE_LIMIT:,}  ({pct_node:.1f}%)")
    print(f"[{status_e}] Edges  {edge_total:>10,} / {EDGE_LIMIT:,}  ({pct_edge:.1f}%)")

    if not node_ok:
        print(f"\n⚠ Node vượt ngưỡng AuraDB Free ({NODE_LIMIT:,})!")
        print("  → Dùng --sample N nhỏ hơn ở transform stage, hoặc --limit-aura khi load.")
    if not edge_ok:
        print(f"\n⚠ Edge vượt ngưỡng AuraDB Free ({EDGE_LIMIT:,})!")
    if node_ok and edge_ok:
        print("\n✓ Trong ngưỡng AuraDB Free — an toàn để load.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ước tính Node/Edge trước khi load Neo4j")
    parser.add_argument("--dir", type=Path, default=TRANSFORMED_DIR, help="Thư mục chứa file transform đã xong")
    args = parser.parse_args()

    if not args.dir.exists():
        print(f"[WARN] Chưa thấy thư mục {args.dir} — chạy --stage transform trước.")
        sys.exit(1)

    stats = estimate(args.dir)
    print_report(stats)


if __name__ == "__main__":
    main()
