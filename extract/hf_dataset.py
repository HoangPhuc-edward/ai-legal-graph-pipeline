"""Tải dữ liệu thô từ th1nhng0/vietnamese-legal-documents (HF Hub).

Cố tình KHÔNG dùng `datasets.load_dataset()` — thư viện này lỗi `ArrowInvalid`
khi gặp cột `content_html` lớn (ép cast large_string -> string thất bại).
Thay vào đó tải trực tiếp file .parquet qua nhánh `refs/convert/parquet`
bằng `huggingface_hub` + đọc bằng `pyarrow.parquet`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from config import HF_DATASET_REPO, RAW_DIR

logger = logging.getLogger(__name__)

CONFIGS = ("metadata", "content", "relationships")
REVISION = "refs/convert/parquet"


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


def download_all(output_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Tải 3 config (metadata/content/relationships), ghi ra parquet cục bộ.

    Trả về dict {config_name: local_path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for config in CONFIGS:
        logger.info("Đang tải config '%s' từ %s ...", config, HF_DATASET_REPO)
        table = _download_config_table(config)
        out_path = output_dir / f"{config}.parquet"
        pq.write_table(table, out_path)
        logger.info("Đã ghi %s (%d dòng) -> %s", config, table.num_rows, out_path)
        result[config] = out_path
    return result


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all()
