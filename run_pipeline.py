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

from config import EMBEDDED_DIR, FILTERED_DIR, RAW_DIR, TRANSFORMED_DIR
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


def stage_transform(sample: int | None, use_llm: bool, workers: int = 1, input_dir: Path | None = None) -> None:
    from extract import hf_dataset
    from transform import pipeline as transform_pipeline

    src = input_dir or RAW_DIR
    tables = hf_dataset.load_local(src)
    transform_pipeline.run(
        metadata_table=tables["metadata"],
        content_table=tables["content"],
        relationships_table=tables["relationships"],
        sample=sample,
        use_llm=use_llm,
        output_dir=TRANSFORMED_DIR,
        workers=workers,
    )


def stage_embed(concurrency: int = 4, rpm: int = 60) -> None:
    """RAM-safe + rate-limited: đọc textunits.jsonl từng dòng, embed theo batch, ghi ngay.

    rpm: giới hạn requests/phút gửi tới Vertex AI — để không bị 429 quota.
         Mặc định 60 RPM = 1 req/s → 60×250 = 15k TextUnit/phút = 900k/hr.
         Giảm xuống --embed-rpm 30 nếu vẫn còn 429.
    concurrency: số request gửi đồng thời. Với rate limiting, tăng concurrency
         giúp pipeline I/O không bị stall nhưng throughput vẫn bị giới hạn bởi rpm.
    """
    from concurrent.futures import Future, ThreadPoolExecutor
    from datetime import datetime, timezone

    from embed import gemini_embedding

    in_path = TRANSFORMED_DIR / "textunits.jsonl"
    out_path = EMBEDDED_DIR / "textunits.jsonl"

    if not in_path.exists():
        logger.warning("Không có TextUnit nào để embed — chạy --stage transform trước.")
        return

    EMBEDDED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = gemini_embedding._get_client()
    except Exception as exc:
        logger.exception("Không khởi tạo được embedding client")
        return

    BATCH = 250  # Vertex AI tối đa 250 inputs/request
    rate_limiter = gemini_embedding.RateLimiter(rpm=rpm)
    logger.info("Embed bắt đầu: batch=%d, concurrency=%d, rpm=%d (%.1f req/s)",
                BATCH, concurrency, rpm, rpm / 60)

    def _embed_one_batch(batch: list[TextUnit]) -> list[TextUnit]:
        rate_limiter.acquire()  # chặn tại đây nếu đang gửi quá nhanh
        to_embed = [tu for tu in batch if tu.accumulated_text.strip()]
        no_text = [tu for tu in batch if not tu.accumulated_text.strip()]
        for tu in no_text:
            tu.error_log = "accumulated_text rỗng — bỏ qua embed"
        if to_embed:
            embeddings = gemini_embedding._embed_batch_with_retry(client, to_embed)
            if embeddings is None:
                error_msg = "Hết lần retry (429 quota) hoặc lỗi không retry được"
                for tu in to_embed:
                    tu.error_log = error_msg
                gemini_embedding._write_embed_errors([tu.unit_id for tu in to_embed], error_msg)
            else:
                now = datetime.now(timezone.utc)
                for tu, emb in zip(to_embed, embeddings):
                    tu.embedding = list(emb.values)
                    tu.embedded_at = now
                    tu.error_log = None
        return no_text + to_embed

    total = succeeded = failed = 0
    embed_batch: list[TextUnit] = []
    active_futures: list[Future] = []

    def _submit():
        if embed_batch:
            snapshot = list(embed_batch)
            embed_batch.clear()
            active_futures.append(executor.submit(_embed_one_batch, snapshot))

    def _drain_one(f_out):
        nonlocal total, succeeded, failed
        fut = active_futures.pop(0)
        for tu in fut.result():
            f_out.write(tu.model_dump_json() + "\n")
            total += 1
            if tu.embedding is not None:
                succeeded += 1
            elif tu.type != "cache_action" and tu.error_log:
                failed += 1
        if total % 50_000 < BATCH:
            logger.info("Embed: %d ghi xong (%d OK, %d lỗi)...", total, succeeded, failed)

    with (
        open(in_path, "r", encoding="utf-8") as f_in,
        open(out_path, "w", encoding="utf-8") as f_out,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            tu = TextUnit.model_validate_json(line)
            if tu.type == "cache_action":
                f_out.write(tu.model_dump_json() + "\n")
                total += 1
                continue
            embed_batch.append(tu)
            if len(embed_batch) >= BATCH:
                _submit()
                if len(active_futures) >= concurrency:
                    _drain_one(f_out)

        _submit()  # batch cuối
        while active_futures:
            _drain_one(f_out)

    logger.info("Embed xong %d TextUnit (%d OK, %d lỗi) -> %s", total, succeeded, failed, out_path)


def _read_load_artifacts():
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

    return norms, components, actions, relations, component_text_units, cache_text_units, component_textunit_map, action_links, text_units


def stage_load(limit_aura: bool = False) -> None:
    from load import loaders
    from load.neo4j_client import Neo4jClient

    (
        norms, components, actions, relations,
        component_text_units, cache_text_units,
        component_textunit_map, action_links, text_units,
    ) = _read_load_artifacts()

    schema_path = Path(__file__).parent / "load" / "schema_init.cypher"

    with Neo4jClient() as client:
        client.run_schema_init(schema_path)
        if limit_aura:
            loaders.load_with_limit(
                client=client,
                norms=norms,
                components=components,
                component_text_units=component_text_units,
                component_textunit_map=component_textunit_map,
                actions=actions,
                action_links=action_links,
                cache_text_units=cache_text_units,
                relations=relations,
            )
        else:
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Số process song song cho Pass 1 transform (mặc định 1 — an toàn cho laptop; Colab dùng 4)",
    )
    parser.add_argument(
        "--limit-aura",
        action="store_true",
        help="Giới hạn load không vượt 200k node / 400k edge (AuraDB Free)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Thư mục parquet input cho transform (mặc định: data/raw/). Dùng data/filtered/ sau khi chạy keyword filter.",
    )
    parser.add_argument(
        "--embed-concurrency",
        type=int,
        default=4,
        help="Số API request embed song song (mặc định 4). Giảm xuống 1 nếu bị 429 liên tục.",
    )
    parser.add_argument(
        "--embed-rpm",
        type=int,
        default=60,
        help=(
            "Giới hạn requests/phút gửi tới Vertex AI embedding (mặc định 60).\n"
            "60 RPM × 250 batch = 15k TextUnit/phút = 900k/hr.\n"
            "Giảm xuống 30 nếu vẫn bị 429. Tăng lên 120-300 nếu quota cho phép."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    use_llm = not args.no_llm

    if args.stage in ("extract", "all"):
        stage_extract()
    if args.stage in ("transform", "all"):
        stage_transform(sample=args.sample, use_llm=use_llm, workers=args.workers, input_dir=args.input_dir)
    if args.stage in ("embed", "all"):
        stage_embed(concurrency=args.embed_concurrency, rpm=args.embed_rpm)
    if args.stage in ("load", "all"):
        stage_load(limit_aura=args.limit_aura)


if __name__ == "__main__":
    main()
