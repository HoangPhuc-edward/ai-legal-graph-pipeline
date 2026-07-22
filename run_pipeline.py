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


def stage_embed(concurrency: int = 4, rpm: int = 60, direct_to_graph: bool = False) -> None:
    """RAM-safe + rate-limited: đọc textunits.jsonl từng dòng, embed theo batch, ghi ngay.

    rpm: giới hạn requests/phút gửi tới Vertex AI — để không bị 429 quota.
         Mặc định 60 RPM = 1 req/s → 60×250 = 15k TextUnit/phút = 900k/hr.
         Giảm xuống --embed-rpm 30 nếu vẫn còn 429.
    concurrency: số request gửi đồng thời. Với rate limiting, tăng concurrency
         giúp pipeline I/O không bị stall nhưng throughput vẫn bị giới hạn bởi rpm.
    direct_to_graph: nếu True, upsert embedding thẳng lên Neo4j thay vì ghi JSONL.
         Yêu cầu --stage load đã chạy trước để TextUnit node tồn tại trong Neo4j.
    """
    from concurrent.futures import Future, ThreadPoolExecutor
    from datetime import datetime, timezone

    from embed import gemini_embedding

    in_path = TRANSFORMED_DIR / "textunits.jsonl"

    if not in_path.exists():
        logger.warning("Không có TextUnit nào để embed — chạy --stage transform trước.")
        return

    if direct_to_graph:
        _stage_embed_direct(in_path, concurrency=concurrency, rpm=rpm)
    else:
        _stage_embed_jsonl(in_path, concurrency=concurrency, rpm=rpm)


def _build_embed_loop(concurrency: int, rpm: int):
    """Trả về (embed_client, rate_limiter, _embed_one_batch) dùng chung cho cả 2 mode."""
    from datetime import datetime, timezone

    from embed import gemini_embedding

    try:
        embed_client = gemini_embedding._get_client()
    except Exception:
        logger.exception("Không khởi tạo được embedding client")
        return None, None, None

    rate_limiter = gemini_embedding.RateLimiter(rpm=rpm)

    def _embed_one_batch(batch: list[TextUnit]) -> list[TextUnit]:
        rate_limiter.acquire()
        to_embed = [tu for tu in batch if tu.accumulated_text.strip()]
        no_text = [tu for tu in batch if not tu.accumulated_text.strip()]
        for tu in no_text:
            tu.error_log = "accumulated_text rỗng — bỏ qua embed"
        if to_embed:
            embeddings = gemini_embedding._embed_batch_with_retry(embed_client, to_embed)
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

    return embed_client, rate_limiter, _embed_one_batch


def _stage_embed_jsonl(in_path: Path, concurrency: int, rpm: int) -> None:
    """Mode mặc định: embed → ghi EMBEDDED_DIR/textunits.jsonl."""
    from concurrent.futures import Future, ThreadPoolExecutor

    out_path = EMBEDDED_DIR / "textunits.jsonl"
    EMBEDDED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Resume: đọc output hiện có, giữ lại unit thành công, retry unit lỗi ---
    # Stream-copy (không giữ toàn bộ file trong RAM) để tránh OOM khi file lớn.
    done_ids: set[str] = set()
    if out_path.exists():
        file_mb = out_path.stat().st_size / 1_048_576
        logger.info(
            "Resume: đang đọc file cũ (%.0f MB) — có thể mất vài phút nếu file lớn...",
            file_mb,
        )
        print(f"Resume: đang đọc {out_path.name} ({file_mb:.0f} MB)...")
        tmp_path = out_path.with_suffix(".tmp")
        retry_count = 0
        lines_read = 0
        with (
            open(out_path, "r", encoding="utf-8") as f_check,
            open(tmp_path, "w", encoding="utf-8") as f_tmp,
        ):
            for line in f_check:
                line = line.strip()
                if not line:
                    continue
                lines_read += 1
                if lines_read % 10_000 == 0:
                    logger.info("Resume: đã đọc %d dòng (%d OK, %d lỗi)...", lines_read, len(done_ids), retry_count)
                    print(f"  ... đã đọc {lines_read:,} dòng ({len(done_ids):,} OK, {retry_count:,} lỗi)")
                try:
                    tu = TextUnit.model_validate_json(line)
                    if tu.embedding is not None or tu.type == "cache_action":
                        done_ids.add(tu.unit_id)
                        f_tmp.write(line + "\n")
                    else:
                        retry_count += 1
                except Exception:
                    pass
        if done_ids or retry_count:
            tmp_path.replace(out_path)  # atomic rename — tránh truncate giữa chừng
            logger.info(
                "Resume: %d đã embed OK (giữ lại), %d lỗi (sẽ retry lại lần này).",
                len(done_ids), retry_count,
            )
        else:
            tmp_path.unlink(missing_ok=True)

    _, _, _embed_one_batch = _build_embed_loop(concurrency, rpm)
    if _embed_one_batch is None:
        return

    BATCH = 250
    logger.info("Embed → JSONL: batch=%d, concurrency=%d, rpm=%d", BATCH, concurrency, rpm)

    total = skipped = succeeded = failed = 0
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

    file_mode = "a" if done_ids else "w"
    with (
        open(in_path, "r", encoding="utf-8") as f_in,
        open(out_path, file_mode, encoding="utf-8") as f_out,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            tu = TextUnit.model_validate_json(line)
            if tu.unit_id in done_ids:
                skipped += 1
                continue
            if tu.type == "cache_action":
                f_out.write(tu.model_dump_json() + "\n")
                total += 1
                continue
            embed_batch.append(tu)
            if len(embed_batch) >= BATCH:
                _submit()
                if len(active_futures) >= concurrency:
                    _drain_one(f_out)

        _submit()
        while active_futures:
            _drain_one(f_out)

    if skipped:
        logger.info("Embed xong %d TextUnit mới (%d OK, %d lỗi) + %d đã có sẵn -> %s",
                    total, succeeded, failed, skipped, out_path)
    else:
        logger.info("Embed xong %d TextUnit (%d OK, %d lỗi) -> %s", total, succeeded, failed, out_path)


def _stage_embed_direct(in_path: Path, concurrency: int, rpm: int) -> None:
    """Mode --embed-direct: embed → upsert thẳng lên Neo4j, không ghi JSONL vector."""
    from concurrent.futures import Future, ThreadPoolExecutor

    from load import loaders
    from load.neo4j_client import Neo4jClient

    _, _, _embed_one_batch = _build_embed_loop(concurrency, rpm)
    if _embed_one_batch is None:
        return

    BATCH = 250

    with (
        open(in_path, "r", encoding="utf-8") as f_in,
        Neo4jClient() as neo4j,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        # --- Resume: hỏi Neo4j unit_id nào đã có embedding ---
        done_ids: set[str] = set()
        try:
            records = neo4j.query(
                "MATCH (t:TextUnit) WHERE t.embedding IS NOT NULL RETURN t.unit_id AS unit_id"
            )
            done_ids = {r["unit_id"] for r in records}
        except Exception:
            logger.exception("Không query được Neo4j để kiểm tra resume — bắt đầu từ đầu.")

        if done_ids:
            logger.info("Resume (direct): %d TextUnit đã có embedding trong Neo4j — sẽ bỏ qua.", len(done_ids))
        logger.info("Embed → Neo4j trực tiếp: batch=%d, concurrency=%d, rpm=%d", BATCH, concurrency, rpm)

        total = skipped = succeeded = failed = 0
        embed_batch: list[TextUnit] = []
        active_futures: list[Future] = []

        def _submit():
            if embed_batch:
                snapshot = list(embed_batch)
                embed_batch.clear()
                active_futures.append(executor.submit(_embed_one_batch, snapshot))

        def _drain_one():
            nonlocal total, succeeded, failed
            fut = active_futures.pop(0)
            batch_result = fut.result()
            rows = [
                {
                    "unit_id": tu.unit_id,
                    "embedding": tu.embedding,
                    "embedded_at": tu.embedded_at.isoformat() if tu.embedded_at else None,
                    "error_log": tu.error_log,
                }
                for tu in batch_result
                if tu.type != "cache_action"  # cache_action: embedding luôn null, không upsert
            ]
            if rows:
                loaders.upsert_textunit_embeddings(neo4j, rows)
            for tu in batch_result:
                total += 1
                if tu.embedding is not None:
                    succeeded += 1
                elif tu.type != "cache_action" and tu.error_log:
                    failed += 1
            if total % 50_000 < BATCH:
                logger.info("Embed direct: %d xong (%d OK, %d lỗi)...", total, succeeded, failed)

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            tu = TextUnit.model_validate_json(line)
            if tu.unit_id in done_ids:
                skipped += 1
                continue
            if tu.type == "cache_action":
                total += 1
                continue  # cache_action: embedding luôn null, không cần upsert
            embed_batch.append(tu)
            if len(embed_batch) >= BATCH:
                _submit()
                if len(active_futures) >= concurrency:
                    _drain_one()

        _submit()
        while active_futures:
            _drain_one()

    if skipped:
        logger.info("Embed direct xong %d TextUnit mới (%d OK, %d lỗi) + %d đã có sẵn.",
                    total, succeeded, failed, skipped)
    else:
        logger.info("Embed direct xong %d TextUnit (%d OK, %d lỗi) → Neo4j.", total, succeeded, failed)


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
    parser = argparse.ArgumentParser(
        description=(
            "Vietnamese Legal Knowledge Graph — ETL pipeline\n"
            "\n"
            "Các stage (chạy theo thứ tự):\n"
            "  extract    Download dataset từ HuggingFace → data/raw/*.parquet\n"
            "  transform  HTML → Component tree + TextUnit → data/transformed/*.jsonl  (2-pass)\n"
            "  embed      TextUnit → vector 768-dim (Gemini) → data/embedded/textunits.jsonl\n"
            "             Có resume: unit đã embed OK sẽ bị bỏ qua, unit lỗi sẽ được retry.\n"
            "  load       JSONL → Neo4j (MERGE — idempotent, an toàn chạy lại)\n"
            "  all        Chạy cả 4 stage theo thứ tự trên\n"
            "\n"
            "Lưu ý dữ liệu file:\n"
            "  transform ghi đè (mode 'w') — chạy lại sẽ xoá output cũ.\n"
            "  embed có resume — chạy lại giữ unit đã embed, chỉ retry unit lỗi.\n"
            "  load dùng MERGE — chạy lại không duplicate, nhưng không xoá node cũ trong Neo4j."
        ),
        epilog=(
            "Ví dụ lệnh thường dùng:\n"
            "\n"
            "  # Kiểm tra môi trường trước khi chạy\n"
            "  python tools/health_check.py --skip-llm\n"
            "\n"
            "  # Test nhanh transform (bắt buộc --sample khi debug)\n"
            "  python run_pipeline.py --stage transform --sample 200 --no-llm\n"
            "\n"
            "  # Transform full corpus, 4 worker (Colab/server)\n"
            "  python run_pipeline.py --stage transform --workers 4\n"
            "\n"
            "  # Embed — tiết kiệm disk, upsert thẳng Neo4j (cần load chạy trước)\n"
            "  python run_pipeline.py --stage embed --embed-direct\n"
            "\n"
            "  # Embed — giảm tốc độ nếu bị 429 quota\n"
            "  python run_pipeline.py --stage embed --embed-rpm 30 --embed-concurrency 1\n"
            "\n"
            "  # Load lên AuraDB Free (giới hạn 200k node / 400k edge)\n"
            "  python run_pipeline.py --stage load --limit-aura\n"
            "\n"
            "  # Chạy toàn bộ pipeline từ đầu\n"
            "  python run_pipeline.py --stage all --workers 4 --embed-rpm 120\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Core ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--stage",
        required=True,
        choices=["extract", "transform", "embed", "load", "all"],
        metavar="{extract,transform,embed,load,all}",
        help="Stage cần chạy (bắt buộc). Xem mô tả phía trên.",
    )

    # ── Transform ─────────────────────────────────────────────────────────────
    g_tr = parser.add_argument_group("Transform  (--stage transform / all)")
    g_tr.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Giới hạn N văn bản đầu tiên — BẮT BUỘC khi debug để tránh xử lý full corpus.",
    )
    g_tr.add_argument(
        "--no-llm",
        action="store_true",
        help="Tắt LLM call ở Pass 2 action_extractor, chỉ dùng regex. Nhanh hơn, không tốn API.",
    )
    g_tr.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Số process song song cho Pass 1 (mặc định: 1).\n"
             "Laptop: giữ 1. Colab/server: 4 là hợp lý.",
    )
    g_tr.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Thư mục chứa *.parquet input (mặc định: data/raw/).\n"
             "Dùng data/filtered/ nếu đã chạy keyword filter trước.",
    )

    # ── Embed ─────────────────────────────────────────────────────────────────
    g_em = parser.add_argument_group(
        "Embed  (--stage embed / all)",
        "Vertex AI Gemini embedding-001, batch 250 TextUnit/request.\n"
        "Resume tự động: unit đã embed OK sẽ bị bỏ qua khi chạy lại.",
    )
    g_em.add_argument(
        "--embed-concurrency",
        type=int,
        default=4,
        metavar="N",
        help="Số request gửi song song (mặc định: 4).\n"
             "Giảm xuống 1 nếu vẫn bị 429 dù đã giảm --embed-rpm.",
    )
    g_em.add_argument(
        "--embed-rpm",
        type=int,
        default=60,
        metavar="N",
        help="Requests/phút tối đa gửi tới Vertex AI (mặc định: 60).\n"
             "60 RPM × 250 batch ≈ 15k TextUnit/phút ≈ 900k/giờ.\n"
             "Giảm xuống 30 nếu bị 429. Tăng lên 120–300 nếu quota cho phép.",
    )
    g_em.add_argument(
        "--embed-direct",
        action="store_true",
        help="Upsert embedding thẳng lên Neo4j, không ghi file JSONL.\n"
             "Tiết kiệm ~17 GB disk (mỗi vector 768 float32 × 100k TextUnit).\n"
             "Yêu cầu: --stage load đã chạy trước để TextUnit node tồn tại trong Neo4j.",
    )

    # ── Load ──────────────────────────────────────────────────────────────────
    g_ld = parser.add_argument_group("Load  (--stage load / all)")
    g_ld.add_argument(
        "--limit-aura",
        action="store_true",
        help="Bật guard AuraDB Free: dừng trước khi vượt 200k node / 400k edge.\n"
             "Không cần flag này nếu dùng Neo4j Desktop hoặc AuraDB trả phí.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    use_llm = not args.no_llm

    if args.stage in ("extract", "all"):
        stage_extract()
    if args.stage in ("transform", "all"):
        stage_transform(sample=args.sample, use_llm=use_llm, workers=args.workers, input_dir=args.input_dir)
    if args.stage in ("embed", "all"):
        stage_embed(concurrency=args.embed_concurrency, rpm=args.embed_rpm, direct_to_graph=args.embed_direct)
    if args.stage in ("load", "all"):
        stage_load(limit_aura=args.limit_aura)


if __name__ == "__main__":
    main()
