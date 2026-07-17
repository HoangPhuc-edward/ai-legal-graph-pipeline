"""Orchestrator: raw (metadata + content + relationships) -> object Schema.

Vì Action giờ là cầu nối THẬT giữa 2 Component thuộc 2 văn bản khác nhau,
pipeline chạy 2 PASS thay vì xử lý từng văn bản độc lập 1 lần:

  PASS 1 — Structure parsing toàn corpus: chạy structure_parser cho MỌI văn
  bản, build Norm/Component/TextUnit, đồng thời build component_index:
  dict[(norm_id, citation_path)] -> comp_id (vd ("ND_34_2016", "Khoản 4 > Điều 38")).

  PASS 2 — Action extraction xuyên văn bản: đọc relationships.parquet, dùng
  component_index từ Pass 1 để khớp Component B trong văn bản đích — KHÔNG
  parse lại văn bản đích.

Cột thật của th1nhng0/vietnamese-legal-documents (verify qua HF API):
  metadata: id, title, so_ky_hieu, ngay_ban_hanh, loai_van_ban, ngay_co_hieu_luc,
            ngay_het_hieu_luc, nguon_thu_thap, ngay_dang_cong_bao, nganh, linh_vuc,
            co_quan_ban_hanh, chuc_danh, nguoi_ky, pham_vi, thong_tin_ap_dung,
            tinh_trang_hieu_luc
  content:  id, content_html
  relationships: doc_id, other_doc_id, relationship
"""
from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc

from config import TRANSFORMED_DIR
from schema.edges import NormRelation
from schema.enums import ComponentLevel
from schema.nodes import Action, Component, Norm, TextUnit
from transform import relation_classifier
from transform.html_to_markdown import convert as html_to_markdown
from transform.structure_parser import parse_structure
from transform.text_accumulator import build_accumulated_text, build_ancestor_chain

logger = logging.getLogger(__name__)


def _norm_from_metadata_row(row: dict) -> Norm:
    return Norm(
        norm_id=str(row["id"]),
        title=row.get("title") or "",
        norm_number=row.get("so_ky_hieu") or "",
        norm_type=row.get("loai_van_ban") or "",
        published_date=row.get("ngay_ban_hanh"),
        valid_from=row.get("ngay_co_hieu_luc"),
        valid_to=row.get("ngay_het_hieu_luc"),
        publisher=row.get("co_quan_ban_hanh"),
        signer=row.get("nguoi_ky"),
        validity_status=row.get("tinh_trang_hieu_luc"),
        sector=row.get("nganh"),
        field=row.get("linh_vuc"),
        updated_at=datetime.now(timezone.utc),
    )


@dataclass
class TransformedDoc:
    norm: Norm
    components: list[Component] = field(default_factory=list)
    text_units: list[TextUnit] = field(default_factory=list)
    # comp_id -> unit_id của TextUnit "noi_dung" gắn vào component đó (chỉ lá)
    component_text_unit: dict[str, str] = field(default_factory=dict)

    @property
    def component_text(self) -> dict[str, str]:
        """comp_id (lá) -> accumulated_text — dùng làm input cho action_extractor."""
        unit_text = {tu.unit_id: tu.accumulated_text for tu in self.text_units}
        return {
            comp_id: unit_text[unit_id]
            for comp_id, unit_id in self.component_text_unit.items()
            if unit_id in unit_text
        }


_MAX_TU_CHARS = 8_000


def _split_long_line(line: str, max_chars: int) -> list[str]:
    """Cắt 1 dòng đơn > max_chars thành các đoạn ≤ max_chars tại word boundary."""
    if len(line) <= max_chars:
        return [line]
    parts = []
    pos = 0
    while pos < len(line):
        end = pos + max_chars
        if end >= len(line):
            parts.append(line[pos:])
            break
        # Tìm khoảng trắng gần nhất để cắt sạch
        cut = line.rfind(" ", pos, end)
        if cut <= pos:
            cut = end  # không tìm thấy space → cắt cứng
        parts.append(line[pos:cut])
        pos = cut + 1 if line[cut:cut+1] == " " else cut
    return parts


def _split_textunit(unit_id: str, text: str) -> list[tuple[str, str]]:
    """Split oversized accumulated_text thành các phần ≤ _MAX_TU_CHARS.

    Không bao giờ cắt giữa bảng prose ("Bảng [...]:" + data rows).
    Nếu 1 bảng lớn hơn _MAX_TU_CHARS thì split theo data row, lặp lại
    dòng "Bảng [...]:" ở đầu mỗi phần.
    Context header "[...]" được lặp lại trong mỗi chunk.
    """
    if len(text) <= _MAX_TU_CHARS:
        return [(unit_id, text)]

    lines = text.split("\n")

    # Tách context header (dòng đầu dạng "[Tên luật > Điều X > ...]")
    header = ""
    body_start = 0
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        header = lines[0]
        body_start = 1
    body_lines = lines[body_start:]

    # Parse body thành segments: ("table", table_header_line, [data_lines]) | ("para", [lines])
    segments: list[tuple] = []
    i = 0
    while i < len(body_lines):
        line = body_lines[i]
        if line.startswith("Bảng ["):
            table_header_line = line
            i += 1
            data_lines: list[str] = []
            # Thu thập data rows — là các dòng có ": " sau tên cột (không phải "Bảng [")
            while i < len(body_lines) and not body_lines[i].startswith("Bảng [") and not body_lines[i].startswith("["):
                if body_lines[i].strip():
                    data_lines.append(body_lines[i])
                i += 1
            segments.append(("table", table_header_line, data_lines))
        else:
            # Para block
            para: list[str] = []
            while i < len(body_lines) and not body_lines[i].startswith("Bảng ["):
                para.append(body_lines[i])
                i += 1
            segments.append(("para", para))

    # Pack segments vào chunks ≤ _MAX_TU_CHARS
    chunks: list[str] = []
    current: list[str] = [header] if header else []
    current_len = len(header) + 1 if header else 0

    def _flush():
        nonlocal current, current_len
        if current and (current != [header] if header else current):
            chunks.append("\n".join(current))
        current = [header] if header else []
        current_len = len(header) + 1 if header else 0

    for seg in segments:
        if seg[0] == "para":
            # Nếu có dòng đơn > _MAX_TU_CHARS (ví dụ HTML convert ra 1 dòng khổng lồ)
            # → expand ra các sub-lines trước khi pack
            para_lines_raw = seg[1]
            para_lines: list[str] = []
            header_overhead = len(header) + 1 if header else 0
            line_max = max(500, _MAX_TU_CHARS - header_overhead)
            for ln in para_lines_raw:
                if len(ln) > line_max:
                    para_lines.extend(_split_long_line(ln, line_max))
                else:
                    para_lines.append(ln)

            for ln in para_lines:
                ln_len = len(ln) + 1
                if current_len + ln_len > _MAX_TU_CHARS and current != ([header] if header else []):
                    _flush()
                current.append(ln)
                current_len += ln_len

        else:  # "table"
            _, tbl_header, data_lines = seg
            tbl_header_len = len(tbl_header) + 1

            if not data_lines:
                # Empty table after prose conversion (shouldn't happen but be safe)
                continue

            full_table_lines = [tbl_header] + data_lines
            full_table_text = "\n".join(full_table_lines)
            full_table_len = len(full_table_text) + 1

            # Cả bảng vừa trong 1 chunk
            if current_len + full_table_len <= _MAX_TU_CHARS:
                current.extend(full_table_lines)
                current_len += full_table_len

            # Bảng không vừa nhưng chunk hiện tại còn nội dung → flush trước
            elif current != ([header] if header else []):
                _flush()
                # Thử thêm vào chunk mới
                if tbl_header_len + full_table_len <= _MAX_TU_CHARS:
                    current.extend(full_table_lines)
                    current_len += full_table_len
                else:
                    # Bảng quá lớn → row-split
                    _split_large_table(header, tbl_header, data_lines, chunks)

            else:
                # Chunk hiện tại trống, bảng vẫn quá lớn → row-split
                _split_large_table(header, tbl_header, data_lines, chunks)

    # Flush chunk cuối
    if current and current != ([header] if header else []):
        chunks.append("\n".join(current))

    if not chunks:
        return [(unit_id, text)]
    if len(chunks) == 1:
        return [(unit_id, chunks[0])]
    return [(f"{unit_id}__p{i + 1}", chunk) for i, chunk in enumerate(chunks)]


def _split_large_table(
    ctx_header: str,
    tbl_header: str,
    data_lines: list[str],
    chunks: list[str],
) -> None:
    """Row-split 1 bảng quá lớn, lặp lại ctx_header + tbl_header ở mỗi chunk."""
    overhead = len(ctx_header) + 1 + len(tbl_header) + 1 if ctx_header else len(tbl_header) + 1
    current: list[str] = [ctx_header, tbl_header] if ctx_header else [tbl_header]
    current_len = overhead

    row_max = max(500, _MAX_TU_CHARS - overhead)
    for row in data_lines:
        # Nếu 1 data row đơn vượt giới hạn → sub-split theo word boundary
        sub_rows = _split_long_line(row, row_max) if len(row) > row_max else [row]
        for sub in sub_rows:
            sub_len = len(sub) + 1
            if current_len + sub_len > _MAX_TU_CHARS and len(current) > (2 if ctx_header else 1):
                chunks.append("\n".join(current))
                current = [ctx_header, tbl_header] if ctx_header else [tbl_header]
                current_len = overhead
            current.append(sub)
            current_len += sub_len

    if len(current) > (2 if ctx_header else 1):
        chunks.append("\n".join(current))


def transform_one(metadata_row: dict, content_html: str) -> TransformedDoc:
    """1 hàng raw -> {Norm, [Component], [TextUnit]} — phần việc của Pass 1
    cho riêng 1 văn bản. Action/NormRelation xử lý ở Pass 2 vì cần
    component_index toàn cục qua nhiều văn bản."""
    norm = _norm_from_metadata_row(metadata_row)
    markdown = html_to_markdown(content_html or "")
    parse_result = parse_structure(norm.norm_id, markdown)

    components_by_id = {c.comp_id: c for c in parse_result.components}
    text_units: list[TextUnit] = []
    component_text_unit: dict[str, str] = {}
    now = datetime.now(timezone.utc)

    for comp_id, raw_text in parse_result.raw_text.items():
        leaf = components_by_id.get(comp_id)
        if leaf is None:
            continue
        ancestor_chain = build_ancestor_chain(leaf, components_by_id)
        accumulated_text = build_accumulated_text(norm, ancestor_chain, raw_text.strip())
        base_unit_id = f"{comp_id}__tu"
        parts = _split_textunit(base_unit_id, accumulated_text)
        for i, (part_unit_id, part_text) in enumerate(parts):
            text_units.append(
                TextUnit(unit_id=part_unit_id, accumulated_text=part_text, type="noi_dung", updated_at=now)
            )
            if i == 0:
                component_text_unit[comp_id] = part_unit_id

    return TransformedDoc(
        norm=norm,
        components=parse_result.components,
        text_units=text_units,
        component_text_unit=component_text_unit,
    )


def _build_component_index_entries(norm_id: str, components: list[Component]) -> dict[tuple[str, str], str]:
    """component_index: dict[(norm_id, citation_path)] -> comp_id.

    citation_path = chuỗi từ chính Component đi lên tới (và bao gồm) Điều tổ
    tiên gần nhất, nối bằng " > ", vd "Khoản 4 > Điều 38". Component ở cấp
    Điều trở lên thì citation_path chỉ gồm chính nó (vd "Điều 38"). Đây đúng
    là cách action_extractor trích citation trong câu văn pháp luật thật
    (Khoản/Điểm luôn đi kèm số Điều vì đánh số lại từ đầu mỗi Điều).
    """
    components_by_id = {c.comp_id: c for c in components}
    index: dict[tuple[str, str], str] = {}

    for c in components:
        path = [c.citation]
        current = c
        if current.level != ComponentLevel.DIEU:
            while current.parent_comp_id is not None:
                parent = components_by_id[current.parent_comp_id]
                path.append(parent.citation)
                if parent.level == ComponentLevel.DIEU:
                    break
                current = parent
        index[(norm_id, " > ".join(path))] = c.comp_id

    return index


async def _transform_relationships_async(
    relationships_table: pa.Table,
    norm_ids: set[str],
    comp_texts_by_norm: dict[str, dict[str, str]],
    norm_numbers: dict[str, str],
    component_index: dict[tuple[str, str], str],
    use_llm: bool = True,
) -> tuple[list[NormRelation], list[tuple[Action, TextUnit, str, str]]]:
    """PASS 2 async — LLM call trong mỗi văn bản chạy đồng thời qua asyncio.gather()
    với Semaphore dùng chung (tối đa MAX_CONCURRENT_LLM request bay cùng lúc).

    Thứ tự xử lý quan hệ vẫn tuần tự (outer for loop), nhưng trong mỗi quan hệ
    tất cả Component cần LLM được gather() song song — giảm latency từ O(N*T)
    xuống O(T) với T là latency 1 LLM call."""
    from transform.action_extractor import MAX_CONCURRENT_LLM

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    def get_component_text_map(norm_id: str) -> dict[str, str]:
        return comp_texts_by_norm.get(norm_id, {})

    def lookup_norm_number(norm_id: str) -> str:
        return norm_numbers.get(norm_id, norm_id)

    known_norm_ids = {norm_id for norm_id, _ in component_index}

    relations: list[NormRelation] = []
    actions: list[tuple[Action, TextUnit, str, str]] = []

    doc_ids_arr = pa.array(list(norm_ids), type=pa.string())
    rel_filtered = relationships_table.filter(
        pc.is_in(
            pc.cast(relationships_table.column("doc_id"), pa.string()),
            value_set=doc_ids_arr,
        )
    )
    rel_rows = rel_filtered.to_pylist()
    total_rels = len(rel_rows)
    logger.info("Pass 2 bắt đầu — %d quan hệ cần xử lý (đã lọc từ corpus)", total_rels)

    for idx, row in enumerate(rel_rows, 1):
        doc_id = str(row["doc_id"])
        other_doc_id = str(row["other_doc_id"])
        label = row["relationship"]
        for item in await relation_classifier.process_relationship_row_async(
            semaphore=semaphore,
            doc_id=doc_id,
            other_doc_id=other_doc_id,
            relationship_label=label,
            component_index=component_index,
            get_component_text_map=get_component_text_map,
            lookup_norm_number=lookup_norm_number,
            use_llm=use_llm,
            known_norm_ids=known_norm_ids,
        ):
            if isinstance(item, NormRelation):
                relations.append(item)
            else:
                actions.append(item)

        if idx % 5000 == 0 or idx == total_rels:
            logger.info(
                "Pass 2: [%d/%d] quan hệ — %d NormRelation, %d Action tích lũy",
                idx, total_rels, len(relations), len(actions),
            )

    return relations, actions


def _transform_batch(pairs: list[tuple[dict, str]]) -> list[TransformedDoc]:
    """Worker Pass 1 — xử lý 1 batch (row, content_html) độc lập trong subprocess.

    Hàm này ở cấp MODULE (không lồng trong hàm khác) vì ProcessPoolExecutor yêu
    cầu có thể pickle được để gửi qua boundary giữa các process (đặc biệt trên
    Windows, dùng spawn context). Mỗi subprocess import lại module hoàn chỉnh
    nên không cần truyền state nào qua argument ngoài data.
    """
    results = []
    for row, content_html in pairs:
        norm_id = str(row["id"])
        try:
            doc = transform_one(row, content_html)
            results.append(doc)
        except Exception as exc:
            logger.exception("Lỗi transform văn bản %s: %s", norm_id, exc)
    return results


def _write_jsonl(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def run(
    metadata_table: pa.Table,
    content_table: pa.Table,
    relationships_table: pa.Table,
    sample: Optional[int] = None,
    use_llm: bool = True,
    output_dir: Path = TRANSFORMED_DIR,
    workers: int = 1,
) -> None:
    """Orchestrate toàn bộ transform stage (2 pass), ghi kết quả ra JSON Lines.

    RAM-safe: Norm/Component/TextUnit được ghi ra đĩa ngay sau mỗi doc (Pass 1)
    thay vì tích lũy trong RAM. Chỉ giữ component_index + comp_texts_by_norm
    (text thuần, không có Pydantic overhead) cho Pass 2.

    workers=1 (mặc định, an toàn cho laptop 8GB): sequential, không spawn process.
    workers>1 (gợi ý Colab: 4): Pass 1 chạy parallel bằng ProcessPoolExecutor —
      mỗi worker xử lý 1 chunk, main process stream-write kết quả từng chunk khi xong.
    """
    if sample is not None:
        metadata_table = metadata_table.slice(0, sample)
    metadata_rows = metadata_table.to_pylist()

    # Index gọn của metadata (không có HTML) — ~22MB cho 47k docs
    meta_by_id: dict[str, dict] = {str(row["id"]): row for row in metadata_rows}
    del metadata_rows
    target_id_set = set(meta_by_id.keys())
    total_docs = len(meta_by_id)

    # Pre-compute tập norm_id có amend relationship → chỉ lưu comp_texts cho những docs này
    # (docs không amend ai thì Pass 2 không cần component_text của chúng)
    amending_norm_ids: set[str] = {
        str(v)
        for v in relationships_table.column("doc_id").to_pylist()
    }

    # Filter content table — giữ dạng Arrow (columnar, ~compact), KHÔNG gọi .to_pylist() toàn bộ
    target_ids_arr = pa.array(list(target_id_set), type=pa.string())
    content_filtered = content_table.filter(
        pc.is_in(pc.cast(content_table.column("id"), pa.string()), value_set=target_ids_arr)
    )

    # ═══════════ PASS 1 — structure parsing + stream-write ngay để giải phóng RAM ═══════════
    # Trong RAM chỉ giữ:
    #   component_index        dict[(norm_id, citation_path) → comp_id]
    #   comp_texts_by_norm     dict[norm_id → dict[comp_id → text]]  — CHỈ cho amending docs
    #   norm_numbers           dict[norm_id → norm_number]
    #   component_textunit_map dict[comp_id → unit_id]
    # Norm/Component/TextUnit objects ghi ra đĩa ngay — không tích lũy.
    component_index: dict[tuple[str, str], str] = {}
    comp_texts_by_norm: dict[str, dict[str, str]] = {}
    norm_numbers: dict[str, str] = {}
    component_textunit_map: dict[str, str] = {}

    logger.info("Pass 1 bắt đầu — %d văn bản cần transform (%d docs có relationship)",
                total_docs, len(amending_norm_ids & target_id_set))
    output_dir.mkdir(parents=True, exist_ok=True)
    n_comp_total = 0
    n_tu_total = 0

    def _process_and_stream(docs: list[TransformedDoc], f_norms, f_comps, f_tu) -> None:
        nonlocal n_comp_total, n_tu_total
        for doc in docs:
            f_norms.write(doc.norm.model_dump_json() + "\n")
            for c in doc.components:
                f_comps.write(c.model_dump_json() + "\n")
            for tu in doc.text_units:
                f_tu.write(tu.model_dump_json() + "\n")
            norm_id = doc.norm.norm_id
            component_index.update(_build_component_index_entries(norm_id, doc.components))
            # Chỉ giữ comp_texts cho docs thực sự có amend relationship (tiết kiệm RAM)
            if norm_id in amending_norm_ids:
                comp_texts_by_norm[norm_id] = doc.component_text
            norm_numbers[norm_id] = doc.norm.norm_number
            component_textunit_map.update(doc.component_text_unit)
            n_comp_total += len(doc.components)
            n_tu_total += len(doc.text_units)

    # Số HTML strings trong Python memory tại một thời điểm
    CONTENT_BATCH = 200

    with (
        open(output_dir / "norms.jsonl", "w", encoding="utf-8") as f_norms,
        open(output_dir / "components.jsonl", "w", encoding="utf-8") as f_comps,
        open(output_dir / "textunits.jsonl", "w", encoding="utf-8") as f_tu,
    ):
        if workers <= 1:
            processed_ids: set[str] = set()
            idx = 0
            # Stream content 200 rows tại một thời điểm — không bao giờ to_pylist() toàn bộ
            for batch in content_filtered.to_batches(max_chunksize=CONTENT_BATCH):
                batch_rows = batch.to_pylist()
                batch_pairs = [
                    (meta_by_id[str(r["id"])], r["content_html"] or "")
                    for r in batch_rows
                    if str(r["id"]) in meta_by_id
                ]
                for r in batch_rows:
                    processed_ids.add(str(r["id"]))
                if batch_pairs:
                    _process_and_stream(_transform_batch(batch_pairs), f_norms, f_comps, f_tu)
                    idx += len(batch_pairs)
                    if idx % 2000 < CONTENT_BATCH or idx >= total_docs:
                        logger.info(
                            "Pass 1: [%d/%d] văn bản — %d Component, %d TextUnit (đã ghi đĩa)",
                            idx, total_docs, n_comp_total, n_tu_total,
                        )
            # Metadata rows không có HTML (scan PDF) — tạo Norm nhưng không có TextUnit
            no_html_ids = target_id_set - processed_ids
            if no_html_ids:
                no_html_pairs = [(meta_by_id[doc_id], "") for doc_id in no_html_ids]
                logger.info("Pass 1: %d văn bản không có HTML", len(no_html_pairs))
                _process_and_stream(_transform_batch(no_html_pairs), f_norms, f_comps, f_tu)
                idx += len(no_html_pairs)
            logger.info(
                "Pass 1 hoàn tất: %d/%d văn bản — %d Component, %d TextUnit",
                idx, total_docs, n_comp_total, n_tu_total,
            )
        else:
            # workers > 1: collect pairs từ content batches, dispatch từng chunk_size docs
            chunk_size = max(CONTENT_BATCH, total_docs // workers)
            logger.info("Pass 1 — %d worker, chunk_size ~%d", workers, chunk_size)
            pending: list[tuple[dict, str]] = []
            processed_ids_w: set[str] = set()
            idx = 0
            chunk_num = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures: dict = {}

                def _submit_chunk(chunk: list) -> None:
                    nonlocal chunk_num
                    futures[executor.submit(_transform_batch, chunk)] = chunk_num
                    chunk_num += 1

                for batch in content_filtered.to_batches(max_chunksize=CONTENT_BATCH):
                    for r in batch.to_pylist():
                        doc_id = str(r["id"])
                        if doc_id in meta_by_id:
                            pending.append((meta_by_id[doc_id], r["content_html"] or ""))
                            processed_ids_w.add(doc_id)
                    while len(pending) >= chunk_size:
                        _submit_chunk(pending[:chunk_size])
                        pending = pending[chunk_size:]

                if pending:
                    _submit_chunk(pending)
                no_html = [(meta_by_id[d], "") for d in target_id_set - processed_ids_w]
                if no_html:
                    _submit_chunk(no_html)

                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        chunk_docs = future.result()
                        _process_and_stream(chunk_docs, f_norms, f_comps, f_tu)
                        idx += len(chunk_docs)
                        logger.info(
                            "Pass 1: chunk %d xong (tổng tích lũy: %d Component, %d TextUnit)",
                            chunk_idx, n_comp_total, n_tu_total,
                        )
                    except Exception:
                        logger.exception("Worker chunk %d lỗi, bỏ qua.", chunk_idx)

        # ═══════════ PASS 2 — action extraction async, LLM concurrent qua Semaphore ═══════════
        relations, actions = asyncio.run(
            _transform_relationships_async(
                relationships_table,
                norm_ids=set(norm_numbers.keys()),
                comp_texts_by_norm=comp_texts_by_norm,
                norm_numbers=norm_numbers,
                component_index=component_index,
                use_llm=use_llm,
            )
        )

        # Append cache TextUnit (type="cache_action") vào textunits.jsonl đã ghi ở Pass 1
        cache_text_units = [tu for _, tu, _, _ in actions]
        for tu in cache_text_units:
            f_tu.write(tu.model_dump_json() + "\n")

    from transform import action_extractor as _ae
    _ae.log_stats()

    action_objs = [a for a, _, _, _ in actions]
    _write_jsonl(output_dir / "actions.jsonl", action_objs)
    _write_jsonl(output_dir / "relations.jsonl", relations)

    with open(output_dir / "component_textunit_map.json", "w", encoding="utf-8") as f:
        json.dump(component_textunit_map, f, ensure_ascii=False, indent=2)

    # action_links.jsonl: 1 dòng/Action — đủ thông tin cho load_action_edges()
    # tạo HAS_ACTION (Component A -> Action), APPLY_TO (Action -> Component B),
    # và HAS_TEXTUNIT (Action -> cache TextUnit) trong cùng 1 bước.
    action_links = [
        {
            "action_id": action.action_id,
            "comp_a_id": comp_a_id,
            "comp_b_id": comp_b_id,
            "cache_unit_id": cache_tu.unit_id,
        }
        for action, cache_tu, comp_a_id, comp_b_id in actions
    ]
    with open(output_dir / "action_links.jsonl", "w", encoding="utf-8") as f:
        for link in action_links:
            f.write(json.dumps(link, ensure_ascii=False) + "\n")

    logger.info(
        "Transform xong (2 pass): %d Norm, %d Component, %d TextUnit (%d cache), %d Action, %d NormRelation",
        len(norm_numbers),
        n_comp_total,
        n_tu_total + len(cache_text_units),
        len(cache_text_units),
        len(action_objs),
        len(relations),
    )