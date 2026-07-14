"""Tải dữ liệu thô từ th1nhng0/vietnamese-legal-documents (HF Hub).

Cố tình KHÔNG dùng `datasets.load_dataset()` — thư viện này lỗi `ArrowInvalid`
khi gặp cột `content_html` lớn (ép cast large_string -> string thất bại).
Thay vào đó tải trực tiếp file .parquet qua nhánh `refs/convert/parquet`
bằng `huggingface_hub` + đọc bằng `pyarrow.parquet`.

CLI:
    python -m extract.hf_dataset extract                          # tải full 3 config về RAW_DIR
    python -m extract.hf_dataset download                         # như extract nhưng linh hoạt hơn (dành cho Colab)
    python -m extract.hf_dataset download --output-dir /content/drive/MyDrive/legal/raw
    python -m extract.hf_dataset download --configs metadata relationships   # chỉ tải 2 config
    python -m extract.hf_dataset download --force                            # tải lại dù đã có
    python -m extract.hf_dataset keyword [--keywords-file F] [--limit N] [--output F]
    python -m extract.hf_dataset sample [--n N] [--output F]

QUAN TRỌNG (xem CLAUDE.md Mục 1): mọi lệnh `keyword`/`sample` ở dưới đọc parquet
THEO BATCH (`pf.iter_batches(batch_size=...)`), KHÔNG bao giờ `pq.read_table()`
nguyên file `content.parquet`/`relationships.parquet` — máy RAM giới hạn, load full
là crash chắc chắn.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from config import DATA_DIR, FILTERED_DIR, HF_DATASET_REPO, RAW_DIR

logger = logging.getLogger(__name__)

CONFIGS = ("metadata", "content", "relationships")
REVISION = "refs/convert/parquet"

DEFAULT_KEYWORDS_PATH = Path(__file__).parent / "keywords.txt"
SAMPLES_DIR = DATA_DIR / "samples"


def _list_parquet_files(config: str) -> list[str]:
    """Liệt kê toàn bộ file parquet (có thể nhiều shard) của 1 config trên nhánh convert."""
    api = HfApi()
    all_files = api.list_repo_files(
        repo_id=HF_DATASET_REPO, repo_type="dataset", revision=REVISION
    )
    return sorted(
        f for f in all_files if f.startswith(f"{config}/") and f.endswith(".parquet")
    )


def _download_config_table(config: str) -> pa.Table:
    files = _list_parquet_files(config)
    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy file parquet nào cho config '{config}' trên {HF_DATASET_REPO}"
        )

    tables = []
    for remote_path in files:
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=remote_path,
            revision=REVISION,
        )
        tables.append(pq.read_table(local_path))

    return pa.concat_tables(tables, promote_options="permissive")


def download_configs(
    configs: list[str] | None = None,
    output_dir: Path = RAW_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Tải 1 hoặc nhiều config từ HF Hub, ghi ra parquet cục bộ.

    configs=None -> tải cả 3 (metadata/content/relationships).
    force=False  -> bỏ qua config đã có file cục bộ (an toàn cho Colab: không
                    tải lại file lớn nếu lỡ re-run cell).
    Trả về dict {config_name: local_path} chỉ gồm những config đã tải/có sẵn.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = list(configs) if configs else list(CONFIGS)
    result: dict[str, Path] = {}
    for config in targets:
        out_path = output_dir / f"{config}.parquet"
        if out_path.exists() and not force:
            logger.info("Đã có %s (%d dòng) — bỏ qua (dùng --force để tải lại).", out_path, pq.ParquetFile(out_path).metadata.num_rows)
            result[config] = out_path
            continue
        logger.info("Đang tải config '%s' từ %s ...", config, HF_DATASET_REPO)
        table = _download_config_table(config)
        pq.write_table(table, out_path)
        logger.info("Đã ghi %s (%d dòng) -> %s", config, table.num_rows, out_path)
        result[config] = out_path
    return result


def download_all(output_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Tải 3 config (metadata/content/relationships), ghi ra parquet cục bộ.

    Trả về dict {config_name: local_path}.
    """
    return download_configs(output_dir=output_dir, force=True)


def load_local(output_dir: Path = RAW_DIR) -> dict[str, pa.Table]:
    """Đọc lại 3 file parquet đã tải về (dùng cho transform stage)."""
    tables = {}
    for config in CONFIGS:
        path = output_dir / f"{config}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Chưa thấy {path} — chạy `--stage extract` trước."
            )
        tables[config] = pq.read_table(path)
    return tables


def _ensure_local_table_path(config: str, raw_dir: Path = RAW_DIR) -> Path:
    """Trả về đường dẫn file parquet cục bộ của `config`, tự tải nếu chưa có.

    Không bao giờ đọc nội dung vào RAM ở đây — chỉ tải file về đĩa nếu thiếu.
    """
    path = raw_dir / f"{config}.parquet"
    if not path.exists():
        logger.info("Chưa có %s cục bộ — tải config '%s' từ %s ...", path, config, HF_DATASET_REPO)
        raw_dir.mkdir(parents=True, exist_ok=True)
        table = _download_config_table(config)
        pq.write_table(table, path)
        logger.info("Đã ghi %s (%d dòng)", path, table.num_rows)
    return path


def _load_keywords(keywords_path: Path) -> list[str]:
    keywords: list[str] = []
    with open(keywords_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keywords.append(line.lower())
    if not keywords:
        raise ValueError(f"Không có từ khoá nào trong {keywords_path}")
    return keywords


def _batch_row_dicts(table: pa.Table) -> list[dict]:
    """Chuyển 1 batch (table nhỏ, đã giới hạn batch_size) -> list[dict] cho dễ thao tác."""
    return table.to_pylist()


def _attach_metadata(matched: dict[int, dict], raw_dir: Path = RAW_DIR, batch_size: int = 2000) -> None:
    """Đọc metadata.parquet theo batch, gắn dict metadata vào từng `matched[id]`."""
    metadata_path = _ensure_local_table_path("metadata", raw_dir)
    pf = pq.ParquetFile(metadata_path)
    remaining = set(matched.keys())
    for batch in pf.iter_batches(batch_size=batch_size):
        if not remaining:
            break
        for row in _batch_row_dicts(pa.Table.from_batches([batch])):
            doc_id = row["id"]
            if doc_id in remaining:
                matched[doc_id]["metadata"] = row
                remaining.discard(doc_id)
    for doc_id in remaining:
        matched[doc_id]["metadata"] = None


def _attach_relationships(
    matched: dict[int, dict],
    raw_dir: Path = RAW_DIR,
    limit: int | None = None,
    batch_size: int = 5000,
) -> None:
    """Đọc relationships.parquet theo batch, gắn list quan hệ liên quan tới mỗi id đã match.

    `limit=None` -> lấy FULL quan hệ liên quan (không giới hạn).
    `limit=N` -> dừng ngay khi đã gắn đủ N quan hệ tổng cộng.
    """
    relationships_path = _ensure_local_table_path("relationships", raw_dir)
    pf = pq.ParquetFile(relationships_path)
    ids = set(matched.keys())
    for doc_id in matched:
        matched[doc_id].setdefault("relationships", [])

    total = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=["doc_id", "other_doc_id", "relationship"]):
        doc_ids = batch.column("doc_id").to_pylist()
        other_ids = batch.column("other_doc_id").to_pylist()
        rels = batch.column("relationship").to_pylist()
        for d, o, r in zip(doc_ids, other_ids, rels):
            if limit is not None and total >= limit:
                return
            if d in ids:
                matched[d]["relationships"].append({"other_doc_id": o, "relationship": r, "direction": "outgoing"})
                total += 1
            elif o in ids:
                matched[o]["relationships"].append({"other_doc_id": d, "relationship": r, "direction": "incoming"})
                total += 1


def search_by_keywords(
    keywords_path: Path = DEFAULT_KEYWORDS_PATH,
    limit: int | None = None,
    raw_dir: Path = RAW_DIR,
    batch_size: int = 2000,
) -> dict[int, dict]:
    """Quét `metadata.parquet` THEO BATCH tìm `title` chứa 1 trong các từ khoá ở `keywords_path`.

    Với mỗi văn bản khớp, gắn thêm `metadata` tương ứng (từ metadata.parquet) và
    `relationships` liên quan (từ relationships.parquet, doc_id hoặc other_doc_id
    trùng id khớp). `limit=None` -> quét full + lấy full quan hệ liên quan;
    `limit=N` -> dừng quét ngay khi đủ N văn bản khớp.
    """
    keywords = _load_keywords(keywords_path)
    metadata_path = _ensure_local_table_path("metadata", raw_dir)

    pf = pq.ParquetFile(metadata_path)
    matched: dict[int, dict] = {}
    for batch in pf.iter_batches(batch_size=batch_size, columns=["id", "title"]):
        ids = batch.column("id").to_pylist()
        titles = batch.column("title").to_pylist()
        for doc_id, title in zip(ids, titles):
            if not title:
                continue
            if any(kw in title.lower() for kw in keywords):
                matched[doc_id] = {"title": title}
                if limit is not None and len(matched) >= limit:
                    break
        if limit is not None and len(matched) >= limit:
            break

    logger.info("Khớp từ khoá (title) %s: %d văn bản", keywords, len(matched))
    if not matched:
        return matched

    _attach_metadata(matched, raw_dir)
    _attach_relationships(matched, raw_dir, limit=None)
    return matched


def filter_to_parquet(
    keywords_path: Path = DEFAULT_KEYWORDS_PATH,
    limit: int | None = None,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = FILTERED_DIR,
    batch_size: int = 500,
) -> dict[str, int]:
    """Lọc 3 file parquet theo từ khoá trong title, ghi ra output_dir dưới dạng parquet.

    Bước 1: quét metadata.parquet.title theo batch → thu thập matched_ids
    Bước 2: lọc metadata.parquet → ghi output_dir/metadata.parquet
    Bước 3: lọc content.parquet   → ghi output_dir/content.parquet
    Bước 4: lọc relationships.parquet (doc_id HOẶC other_doc_id trong matched_ids)
             → ghi output_dir/relationships.parquet

    Không bao giờ load full file vào RAM — toàn bộ dùng iter_batches + ParquetWriter.
    Trả về dict {config: số_dòng} để báo cáo.
    """
    keywords = _load_keywords(keywords_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bước 1 — quét metadata.title → matched_ids (nhẹ hơn content.parquet, chính xác hơn)
    meta_path = _ensure_local_table_path("metadata", raw_dir)
    pf = pq.ParquetFile(meta_path)
    matched_ids: set[int] = set()
    for batch in pf.iter_batches(batch_size=batch_size, columns=["id", "title"]):
        ids = batch.column("id").to_pylist()
        titles = batch.column("title").to_pylist()
        for doc_id, title in zip(ids, titles):
            if title and any(kw in title.lower() for kw in keywords):
                matched_ids.add(doc_id)
                if limit is not None and len(matched_ids) >= limit:
                    break
        if limit is not None and len(matched_ids) >= limit:
            break

    logger.info("Khớp từ khoá (title) %s: %d văn bản", keywords, len(matched_ids))
    if not matched_ids:
        logger.warning("Không có văn bản nào khớp từ khoá — không ghi file.")
        return {}

    # value_set dùng chung cho is_in — cast về string vì id có thể lưu dạng large_string
    matched_arr = pa.array([str(x) for x in matched_ids], type=pa.string())
    counts: dict[str, int] = {}

    def _filter_and_write(src_path: Path, out_path: Path, id_columns: list[str]) -> int:
        """Đọc src_path theo batch, giữ dòng có ít nhất 1 id_column nằm trong matched_arr."""
        writer = None
        n = 0
        try:
            pf2 = pq.ParquetFile(src_path)
            for batch in pf2.iter_batches(batch_size=batch_size if "content" in src_path.name else 5000):
                tbl = pa.Table.from_batches([batch])
                mask = None
                for col in id_columns:
                    m = pc.is_in(pc.cast(tbl.column(col), pa.string()), value_set=matched_arr)
                    mask = m if mask is None else pc.or_(mask, m)
                filtered = tbl.filter(mask)
                if filtered.num_rows == 0:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(out_path, filtered.schema)
                writer.write_table(filtered)
                n += filtered.num_rows
        finally:
            if writer:
                writer.close()
        return n

    # Bước 2 — lọc metadata
    meta_path = _ensure_local_table_path("metadata", raw_dir)
    counts["metadata"] = _filter_and_write(meta_path, output_dir / "metadata.parquet", ["id"])
    logger.info("metadata.parquet: %d dòng → %s", counts["metadata"], output_dir / "metadata.parquet")

    # Bước 3 — lọc content
    content_path = _ensure_local_table_path("content", raw_dir)
    counts["content"] = _filter_and_write(content_path, output_dir / "content.parquet", ["id"])
    logger.info("content.parquet: %d dòng → %s", counts["content"], output_dir / "content.parquet")

    # Bước 4 — lọc relationships (cả 2 chiều)
    rel_path = _ensure_local_table_path("relationships", raw_dir)
    counts["relationships"] = _filter_and_write(rel_path, output_dir / "relationships.parquet", ["doc_id", "other_doc_id"])
    logger.info("relationships.parquet: %d dòng → %s", counts["relationships"], output_dir / "relationships.parquet")

    return counts


def sample_data(n: int = 100, raw_dir: Path = RAW_DIR, batch_size: int = 500) -> dict[int, dict]:
    """Lấy mẫu an toàn để chạy thử pipeline: N metadata đầu tiên (mặc định 100) ->
    content tương ứng -> tối đa N quan hệ liên quan tới N văn bản đó.

    Đọc cả 3 file THEO BATCH, dừng ngay khi đủ N dòng — không bao giờ load full file.
    """
    metadata_path = _ensure_local_table_path("metadata", raw_dir)
    pf = pq.ParquetFile(metadata_path)
    sampled: dict[int, dict] = {}
    for batch in pf.iter_batches(batch_size=min(batch_size, n)):
        for row in _batch_row_dicts(pa.Table.from_batches([batch])):
            if len(sampled) >= n:
                break
            sampled[row["id"]] = {"metadata": row, "content_html": None, "relationships": []}
        if len(sampled) >= n:
            break

    logger.info("Lấy mẫu %d metadata", len(sampled))

    content_path = _ensure_local_table_path("content", raw_dir)
    pf_c = pq.ParquetFile(content_path)
    remaining = set(sampled.keys())
    for batch in pf_c.iter_batches(batch_size=batch_size, columns=["id", "content_html"]):
        if not remaining:
            break
        ids = batch.column("id").to_pylist()
        htmls = batch.column("content_html").to_pylist()
        for doc_id, html in zip(ids, htmls):
            if doc_id in remaining:
                sampled[doc_id]["content_html"] = html
                remaining.discard(doc_id)

    n_with_content = sum(1 for v in sampled.values() if v["content_html"] is not None)
    logger.info("Có content tương ứng: %d/%d", n_with_content, len(sampled))

    _attach_relationships(sampled, raw_dir, limit=n)
    return sampled


def _write_result(result: dict[int, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in result.items()}, f, ensure_ascii=False, indent=2)
    logger.info("Đã ghi %d văn bản -> %s", len(result), output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="extract.hf_dataset CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("extract", help="Tải full 3 config (metadata/content/relationships) về data/raw/ (bỏ qua --output-dir)")

    p_download = sub.add_parser(
        "download",
        help="Tải parquet về thư mục tuỳ chọn — dành cho Colab (Google Drive) hoặc máy ngoài",
    )
    p_download.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help="Thư mục đích (mặc định: data/raw/). Colab: /content/drive/MyDrive/legal/raw",
    )
    p_download.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=None,
        metavar="CONFIG",
        help="Config cần tải (metadata/content/relationships). Mặc định: tải cả 3.",
    )
    p_download.add_argument(
        "--force",
        action="store_true",
        help="Tải lại dù file đã tồn tại cục bộ.",
    )

    p_keyword = sub.add_parser(
        "keyword",
        help="Lọc 3 parquet theo từ khoá trong keywords.txt → ghi ra data/filtered/ để dùng với --stage transform --input-dir data/filtered",
    )
    p_keyword.add_argument("--keywords-file", type=Path, default=DEFAULT_KEYWORDS_PATH)
    p_keyword.add_argument("--limit", type=int, default=None, help="Giới hạn số văn bản khớp (mặc định: lấy full)")
    p_keyword.add_argument("--output-dir", type=Path, default=FILTERED_DIR, help="Thư mục ghi parquet đã lọc (mặc định: data/filtered/)")

    p_sample = sub.add_parser(
        "sample",
        help="Lấy mẫu N metadata + content tương ứng + tối đa N relationships liên quan (mặc định N=100)",
    )
    p_sample.add_argument("--n", type=int, default=100)
    p_sample.add_argument("--output", type=Path, default=SAMPLES_DIR / "sample.json")

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()

    if args.command == "extract":
        download_all()
    elif args.command == "download":
        paths = download_configs(
            configs=args.configs,
            output_dir=args.output_dir,
            force=args.force,
        )
        print(f"\nĐã có {len(paths)} file:")
        for cfg, p in paths.items():
            rows = pq.ParquetFile(p).metadata.num_rows
            print(f"  {cfg:20s} {rows:>10,} dòng  ->  {p}")
    elif args.command == "keyword":
        counts = filter_to_parquet(
            keywords_path=args.keywords_file,
            limit=args.limit,
            output_dir=args.output_dir,
        )
        if counts:
            print(f"\nĐã ghi parquet đã lọc vào {args.output_dir}:")
            for cfg, n in counts.items():
                print(f"  {cfg:20s} {n:>10,} dòng")
            print(f"\nChạy tiếp:")
            print(f"  python run_pipeline.py --stage transform --input-dir {args.output_dir}")
    elif args.command == "sample":
        result = sample_data(n=args.n)
        _write_result(result, args.output)


if __name__ == "__main__":
    main()
