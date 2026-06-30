"""Batch embed TextUnit.accumulated_text qua Vertex AI (gemini-embedding-001).

Chạy SAU transform, TRƯỚC load — chỉ cần TextUnit.accumulated_text đã có sẵn
trong bộ nhớ/file tạm, chưa cần đẩy lên Neo4j. Ghi kết quả ngược vào field
`embedding` + `embedded_at`; dòng nào lỗi (timeout, nội dung rỗng) ghi vào
`error_log`, không chặn cả batch.

TextUnit type="cache_action" (sở hữu bởi Action) bị LỌC BỎ khỏi batch embed —
đây là bản sao y nguyên nội dung của TextUnit Component A đã được embed rồi,
embed thêm lần nữa chỉ tốn tiền vô ích (không ai vector-search trực tiếp tới
Action, chỉ tới được qua graph traversal từ Component đã tìm thấy trước đó).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from config import EMBEDDING_MODEL, GCP_LOCATION, GCP_PROJECT
from schema.nodes import TextUnit

logger = logging.getLogger(__name__)

# Vertex AI text-embedding endpoints chấp nhận tối đa ~250 input/request — giữ
# margin an toàn để tránh lỗi payload quá lớn.
EMBED_REQUEST_BATCH_SIZE = 100


def _get_model():
    import vertexai
    from vertexai.language_models import TextEmbeddingModel

    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    return TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_text_units(
    text_units: list[TextUnit], batch_size: int = EMBED_REQUEST_BATCH_SIZE
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
        model = _get_model()
    except Exception as exc:
        logger.exception("Không khởi tạo được model embedding %s", EMBEDDING_MODEL)
        for tu in pending:
            tu.error_log = f"Lỗi khởi tạo model: {exc}"
        return text_units

    now = datetime.now(timezone.utc)
    for batch in _chunks(pending, batch_size):
        texts = [tu.accumulated_text for tu in batch]
        try:
            embeddings = model.get_embeddings(texts)
        except Exception as exc:
            logger.exception("Lỗi embed batch %d item", len(batch))
            for tu in batch:
                tu.error_log = f"Lỗi embed: {exc}"
            continue

        for tu, emb in zip(batch, embeddings):
            tu.embedding = list(emb.values)
            tu.embedded_at = now
            tu.error_log = None

    return text_units
