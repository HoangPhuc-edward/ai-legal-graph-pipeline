"""Batch embed TextUnit.accumulated_text qua Vertex AI (gemini-embedding-001).

Chạy SAU transform, TRƯỚC load — chỉ cần TextUnit.accumulated_text đã có sẵn
trong bộ nhớ/file tạm, chưa cần đẩy lên Neo4j. Ghi kết quả ngược vào field
`embedding` + `embedded_at`; dòng nào lỗi (timeout, nội dung rỗng) ghi vào
`error_log`, không chặn cả batch.

TextUnit type="cache_action" (sở hữu bởi Action) bị LỌC BỎ khỏi batch embed —
đây là bản sao y nguyên nội dung của TextUnit Component A đã được embed rồi,
embed thêm lần nữa chỉ tốn tiền vô ích (không ai vector-search trực tiếp tới
Action, chỉ tới được qua graph traversal từ Component đã tìm thấy trước đó).

Retry policy: chỉ retry khi lỗi 429 ResourceExhausted (quota/phút). Bắt đầu
chờ 60 giây (cần đủ 1 phút để quota reset), nhân đôi mỗi lần, tối đa 5 lần.
Sau 5 lần vẫn lỗi: ghi unit_id vào embed_errors.jsonl, tiếp tục batch tiếp theo.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Generator, Iterable

from config import EMBEDDING_MODEL, GCP_LOCATION, GCP_PROJECT, TRANSFORMED_DIR
from schema.nodes import TextUnit

logger = logging.getLogger(__name__)

# Vertex AI text-embedding endpoints chấp nhận tối đa ~250 input/request — giữ
# margin an toàn để tránh lỗi payload quá lớn.
EMBED_REQUEST_BATCH_SIZE = 100

_MAX_RETRIES = 5
_INITIAL_WAIT_SEC = 60  # quota reset theo phút — không dùng giá trị < 60

EMBED_ERRORS_FILE = TRANSFORMED_DIR / "embed_errors.jsonl"


def _get_client():
    from google import genai

    return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _is_quota_error(exc: Exception) -> bool:
    try:
        from google.api_core.exceptions import ResourceExhausted
        return isinstance(exc, ResourceExhausted)
    except ImportError:
        return "429" in str(exc) or "ResourceExhausted" in type(exc).__name__


def _write_embed_errors(unit_ids: list[str], error: str) -> None:
    EMBED_ERRORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(EMBED_ERRORS_FILE, "a", encoding="utf-8") as f:
        for uid in unit_ids:
            f.write(json.dumps({"unit_id": uid, "error": error, "timestamp": timestamp}, ensure_ascii=False) + "\n")


def _embed_batch_with_retry(client, batch: list[TextUnit]) -> list | None:
    """Thử embed 1 batch, retry khi 429. Trả None nếu hết lần retry."""
    texts = [tu.accumulated_text for tu in batch]
    wait = _INITIAL_WAIT_SEC
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
            return result.embeddings
        except Exception as exc:
            if _is_quota_error(exc):
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "429 quota exceeded — lần %d/%d, chờ %ds rồi thử lại.",
                        attempt, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    wait *= 2
                else:
                    logger.error(
                        "429 quota exceeded — đã thử %d lần, ghi lỗi và bỏ qua batch %d item.",
                        _MAX_RETRIES, len(batch),
                    )
                    return None
            else:
                logger.exception("Lỗi embed batch %d item (không retry)", len(batch))
                return None
    return None


def embed_text_units(
    text_units: list[TextUnit],
    batch_size: int = EMBED_REQUEST_BATCH_SIZE,
) -> list[TextUnit]:
    """Embed TextUnit type="noi_dung" theo batch — BỎ QUA type="cache_action"
    (bản sao y nguyên nội dung đã embed ở nơi khác, xem docstring module).

    Cập nhật in-place embedding/embedded_at/error_log trên từng object, trả về
    cùng list đã cập nhật (kể cả các item bị bỏ qua, giữ nguyên embedding=None).
    """
    embeddable = [tu for tu in text_units if tu.type != "cache_action"]
    pending = [tu for tu in embeddable if tu.embedding is None and tu.accumulated_text.strip()]
    empty = [tu for tu in embeddable if not tu.accumulated_text.strip()]
    for tu in empty:
        tu.error_log = "accumulated_text rỗng — bỏ qua embed"

    if not pending:
        return text_units

    try:
        client = _get_client()
    except Exception as exc:
        logger.exception("Không khởi tạo được client embedding %s", EMBEDDING_MODEL)
        for tu in pending:
            tu.error_log = f"Lỗi khởi tạo client: {exc}"
        return text_units

    now = datetime.now(timezone.utc)
    succeeded = 0
    failed = 0

    for batch in _chunks(pending, batch_size):
        embeddings = _embed_batch_with_retry(client, batch)
        if embeddings is None:
            error_msg = "Hết lần retry (429 quota) hoặc lỗi không retry được"
            for tu in batch:
                tu.error_log = error_msg
            _write_embed_errors([tu.unit_id for tu in batch], error_msg)
            failed += len(batch)
            continue

        for tu, emb in zip(batch, embeddings):
            tu.embedding = list(emb.values)
            tu.embedded_at = now
            tu.error_log = None
        succeeded += len(batch)

    if failed:
        logger.warning(
            "Embed xong: %d thành công, %d thất bại — xem %s để retry.",
            succeeded, failed, EMBED_ERRORS_FILE,
        )
        print(f"Embed xong: {succeeded} thành công, {failed} thất bại — xem {EMBED_ERRORS_FILE} để retry")
    else:
        logger.info("Embed xong: %d thành công, 0 thất bại.", succeeded)

    return text_units


def embed_stream(
    text_units: Iterable[TextUnit],
    batch_size: int = EMBED_REQUEST_BATCH_SIZE,
) -> Generator[TextUnit, None, None]:
    """RAM-safe generator: yield từng TextUnit đã embed, xử lý theo batch.

    Chỉ giữ tối đa `batch_size` TextUnit trong Python memory tại một thời điểm.
    type="cache_action" được yield ngay không cần embed.
    Caller nên ghi mỗi TextUnit ra file ngay sau khi nhận để giải phóng RAM.
    """
    try:
        client = _get_client()
    except Exception as exc:
        logger.exception("Không khởi tạo được embedding client")
        for tu in text_units:
            tu.error_log = f"Lỗi khởi tạo client: {exc}"
            yield tu
        return

    batch: list[TextUnit] = []
    succeeded = 0
    failed = 0

    def _flush_batch() -> list[TextUnit]:
        nonlocal succeeded, failed
        to_embed = [tu for tu in batch if tu.accumulated_text.strip()]
        no_text = [tu for tu in batch if not tu.accumulated_text.strip()]
        for tu in no_text:
            tu.error_log = "accumulated_text rỗng — bỏ qua embed"

        if to_embed:
            embeddings = _embed_batch_with_retry(client, to_embed)
            if embeddings is None:
                error_msg = "Hết lần retry (429 quota) hoặc lỗi không retry được"
                for tu in to_embed:
                    tu.error_log = error_msg
                _write_embed_errors([tu.unit_id for tu in to_embed], error_msg)
                failed += len(to_embed)
            else:
                now = datetime.now(timezone.utc)
                for tu, emb in zip(to_embed, embeddings):
                    tu.embedding = list(emb.values)
                    tu.embedded_at = now
                    tu.error_log = None
                succeeded += len(to_embed)

        result = no_text + to_embed
        batch.clear()
        return result

    for tu in text_units:
        if tu.type == "cache_action":
            yield tu
            continue
        batch.append(tu)
        if len(batch) >= batch_size:
            yield from _flush_batch()

    if batch:
        yield from _flush_batch()

    if failed:
        logger.warning(
            "embed_stream xong: %d thành công, %d thất bại — xem %s",
            succeeded, failed, EMBED_ERRORS_FILE,
        )
    else:
        logger.info("embed_stream xong: %d thành công", succeeded)
