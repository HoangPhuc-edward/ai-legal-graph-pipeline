"""Retry embed cho những TextUnit bị lỗi 429 ở lần chạy trước.

Đọc embed_errors.jsonl (ghi bởi embed/gemini_embedding.py), load textunits.jsonl
để lấy lại accumulated_text, thử embed lại đúng những unit_id bị lỗi, ghi
embedding ngược vào textunits.jsonl (overwrite toàn file).

Chạy:
    python tools/retry_failed_embeddings.py
    python tools/retry_failed_embeddings.py --errors-file path/to/embed_errors.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry embed cho unit_id bị lỗi 429")
    parser.add_argument(
        "--errors-file",
        type=Path,
        default=None,
        help="Path tới embed_errors.jsonl (mặc định: data/transformed/embed_errors.jsonl)",
    )
    args = parser.parse_args()

    from config import TRANSFORMED_DIR
    from schema.nodes import TextUnit
    from embed.gemini_embedding import embed_text_units, EMBED_ERRORS_FILE

    errors_file: Path = args.errors_file or EMBED_ERRORS_FILE
    textunits_file = TRANSFORMED_DIR / "textunits.jsonl"

    if not errors_file.exists():
        print(f"Không tìm thấy {errors_file} — không có gì cần retry.")
        sys.exit(0)

    if not textunits_file.exists():
        print(f"Không tìm thấy {textunits_file} — chạy --stage transform trước.")
        sys.exit(1)

    # Đọc danh sách unit_id cần retry (deduplicate)
    failed_ids: set[str] = set()
    with open(errors_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            failed_ids.add(entry["unit_id"])

    if not failed_ids:
        print("embed_errors.jsonl rỗng — không có gì cần retry.")
        sys.exit(0)

    logger.info("Cần retry %d unit_id.", len(failed_ids))

    # Load toàn bộ TextUnit từ file
    all_units: list[TextUnit] = []
    with open(textunits_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_units.append(TextUnit.model_validate_json(line))

    units_by_id = {tu.unit_id: tu for tu in all_units}
    to_retry = [units_by_id[uid] for uid in failed_ids if uid in units_by_id]
    missing = failed_ids - set(units_by_id)

    if missing:
        logger.warning("%d unit_id trong errors không tìm thấy trong textunits.jsonl: %s", len(missing), missing)

    if not to_retry:
        print("Không có TextUnit nào để retry.")
        sys.exit(0)

    # Reset embedding để embed_text_units không bỏ qua
    for tu in to_retry:
        tu.embedding = None
        tu.error_log = None

    logger.info("Bắt đầu retry embed %d TextUnit...", len(to_retry))
    embed_text_units(to_retry)

    succeeded = [tu for tu in to_retry if tu.embedding is not None]
    still_failed = [tu for tu in to_retry if tu.embedding is None]
    logger.info("Retry xong: %d thành công, %d vẫn lỗi.", len(succeeded), len(still_failed))

    # Ghi lại toàn bộ textunits.jsonl với embedding đã cập nhật
    with open(textunits_file, "w", encoding="utf-8") as f:
        for tu in all_units:
            f.write(tu.model_dump_json() + "\n")
    logger.info("Đã ghi lại %s.", textunits_file)

    # Xoá embed_errors.jsonl nếu không còn lỗi, hoặc ghi lại chỉ những cái vẫn lỗi
    if not still_failed:
        errors_file.unlink()
        print(f"Retry thành công toàn bộ — đã xoá {errors_file}.")
    else:
        still_failed_ids = {tu.unit_id for tu in still_failed}
        remaining_entries = []
        with open(errors_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry["unit_id"] in still_failed_ids:
                    remaining_entries.append(line)
        with open(errors_file, "w", encoding="utf-8") as f:
            f.write("\n".join(remaining_entries) + "\n")
        print(f"Retry xong: {len(succeeded)} thành công, {len(still_failed)} vẫn lỗi — xem {errors_file}.")


if __name__ == "__main__":
    main()
