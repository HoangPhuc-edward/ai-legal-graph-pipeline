"""CLI cho legal_graph_pipeline.

    python run_pipeline.py --stage extract
    python run_pipeline.py --stage transform --sample 200      # test trước khi full
    python run_pipeline.py --stage embed
    python run_pipeline.py --stage load
    python run_pipeline.py --stage all

Mỗi --stage đọc/ghi qua file trung gian (parquet hoặc JSON lines) giữa các
bước, để có thể dừng/chạy lại từng giai đoạn riêng mà không phải chạy lại từ
đầu — quan trọng vì bước embed tốn phí, không muốn re-run khi chỉ đang debug load.

Không chứa logic nghiệp vụ — mọi thuật toán nằm trong transform/, embed/, load/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import EMBEDDED_DIR, RAW_DIR, TRANSFORMED_DIR
from schema.edges import NormRelation
from schema.nodes import Action, Component, Norm, TextUnit

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path, model) -> list:
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(model.model_validate_json(line))
    return items


def _write_jsonl(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def stage_extract() -> None:
    from extract import hf_dataset

    hf_dataset.download_all()


def stage_transform(sample: int | None, use_llm: bool) -> None:
    from extract import hf_dataset
    from transform import pipeline as transform_pipeline

    tables = hf_dataset.load_local(RAW_DIR)
    transform_pipeline.run(
        metadata_table=tables["metadata"],
        content_table=tables["content"],
        relationships_table=tables["relationships"],
        sample=sample,
        use_llm=use_llm,
        output_dir=TRANSFORMED_DIR,
    )


def stage_embed() -> None:
    from embed import gemini_embedding

    text_units = _read_jsonl(TRANSFORMED_DIR / "textunits.jsonl", TextUnit)
    if not text_units:
        logger.warning("Không có TextUnit nào để embed — chạy --stage transform trước.")
        return
    gemini_embedding.embed_text_units(text_units)
    _write_jsonl(EMBEDDED_DIR / "textunits.jsonl", text_units)
    logger.info("Embed xong %d TextUnit -> %s", len(text_units), EMBEDDED_DIR / "textunits.jsonl")


def stage_load() -> None:
    from load import loaders
    from load.neo4j_client import Neo4jClient

    norms = _read_jsonl(TRANSFORMED_DIR / "norms.jsonl", Norm)
    components = _read_jsonl(TRANSFORMED_DIR / "components.jsonl", Component)
    actions = _read_jsonl(TRANSFORMED_DIR / "actions.jsonl", Action)
    relations = _read_jsonl(TRANSFORMED_DIR / "relations.jsonl", NormRelation)

    embedded_path = EMBEDDED_DIR / "textunits.jsonl"
    text_units_path = embedded_path if embedded_path.exists() else TRANSFORMED_DIR / "textunits.jsonl"
    text_units = _read_jsonl(text_units_path, TextUnit)
    component_text_units = [tu for tu in text_units if tu.type != "cache_action"]
    cache_text_units = {tu.unit_id: tu for tu in text_units if tu.type == "cache_action"}

    with open(TRANSFORMED_DIR / "component_textunit_map.json", "r", encoding="utf-8") as f:
        component_textunit_map = json.load(f)

    action_links = []
    action_links_path = TRANSFORMED_DIR / "action_links.jsonl"
    if action_links_path.exists():
        with open(action_links_path, "r", encoding="utf-8") as f:
            action_links = [json.loads(line) for line in f if line.strip()]

    with Neo4jClient() as client:
        client.run_schema_init(Path(__file__).parent / "load" / "schema_init.cypher")
        loaders.load_norms(client, norms)
        loaders.load_components(client, components)
        loaders.load_component_textunits(client, component_text_units, component_textunit_map)
        loaders.load_actions(client, actions)
        loaders.load_action_edges(client, action_links, cache_text_units)
        loaders.load_relations(client, relations)

    logger.info(
        "Load xong: %d Norm, %d Component, %d TextUnit (%d cache), %d Action, %d NormRelation",
        len(norms),
        len(components),
        len(text_units),
        len(cache_text_units),
        len(actions),
        len(relations),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese Legal GraphRAG ETL pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["extract", "transform", "embed", "load", "all"],
    )
    parser.add_argument("--sample", type=int, default=None, help="Giới hạn số văn bản (transform stage)")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Tắt LLM fallback ở action_extractor (chỉ dùng regex Tầng B)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    use_llm = not args.no_llm

    if args.stage in ("extract", "all"):
        stage_extract()
    if args.stage in ("transform", "all"):
        stage_transform(sample=args.sample, use_llm=use_llm)
    if args.stage in ("embed", "all"):
        stage_embed()
    if args.stage in ("load", "all"):
        stage_load()


if __name__ == "__main__":
    main()
