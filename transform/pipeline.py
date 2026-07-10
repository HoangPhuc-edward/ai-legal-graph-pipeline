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
        unit_id = f"{comp_id}__tu"
        text_units.append(
            TextUnit(unit_id=unit_id, accumulated_text=accumulated_text, type="noi_dung", updated_at=now)
        )
        component_text_unit[comp_id] = unit_id

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
    # Fix 4: slice Arrow table TRƯỚC khi convert — tránh materialize 153k dòng rồi bỏ đi
    if sample is not None:
        metadata_table = metadata_table.slice(0, sample)
    metadata_rows = metadata_table.to_pylist()

    # Fix 1: filter content chỉ lấy id cần thiết — tránh materialize toàn bộ 178k HTML.
    # Cast cả column lẫn value_set về pa.string() vì parquet lưu id dạng large_string.
    target_ids = pa.array([str(row["id"]) for row in metadata_rows], type=pa.string())
    content_by_id = {
        row["id"]: row["content_html"]
        for row in content_table.filter(
            pc.is_in(
                pc.cast(content_table.column("id"), pa.string()),
                value_set=target_ids,
            )
        ).to_pylist()
    }

    # ═══════════ PASS 1 — structure parsing + stream-write ngay để giải phóng RAM ═══════════
    # Chỉ giữ trong RAM những gì Pass 2 cần:
    #   component_index        dict[(norm_id, citation_path) → comp_id]   ~compact strings
    #   comp_texts_by_norm     dict[norm_id → dict[comp_id → text]]        ~thay thế doc.component_text
    #   norm_numbers           dict[norm_id → norm_number]                 ~tiny
    #   component_textunit_map dict[comp_id → unit_id]                     ~compact strings
    # Norm/Component/TextUnit objects được ghi ra đĩa ngay — không tích lũy trong RAM.
    component_index: dict[tuple[str, str], str] = {}
    comp_texts_by_norm: dict[str, dict[str, str]] = {}
    norm_numbers: dict[str, str] = {}
    component_textunit_map: dict[str, str] = {}

    pairs = [
        (row, content_by_id.get(row["id"]) or content_by_id.get(str(row["id"])) or "")
        for row in metadata_rows
    ]
    del content_by_id  # HTML thô không cần nữa sau khi build pairs → giải phóng RAM sớm
    total_docs = len(pairs)
    logger.info("Pass 1 bắt đầu — %d văn bản cần transform", total_docs)

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
            comp_texts_by_norm[norm_id] = doc.component_text
            norm_numbers[norm_id] = doc.norm.norm_number
            component_textunit_map.update(doc.component_text_unit)
            n_comp_total += len(doc.components)
            n_tu_total += len(doc.text_units)

    with (
        open(output_dir / "norms.jsonl", "w", encoding="utf-8") as f_norms,
        open(output_dir / "components.jsonl", "w", encoding="utf-8") as f_comps,
        open(output_dir / "textunits.jsonl", "w", encoding="utf-8") as f_tu,
    ):
        if workers <= 1:
            for idx, pair in enumerate(pairs, 1):
                _process_and_stream(_transform_batch([pair]), f_norms, f_comps, f_tu)
                if idx % 500 == 0 or idx == total_docs:
                    logger.info(
                        "Pass 1: [%d/%d] văn bản — %d Component, %d TextUnit (đã ghi đĩa)",
                        idx, total_docs, n_comp_total, n_tu_total,
                    )
        else:
            chunk_size = max(1, total_docs // workers)
            chunks = [pairs[i : i + chunk_size] for i in range(0, total_docs, chunk_size)]
            logger.info("Pass 1 — %d worker, %d chunk (~%d văn bản/chunk)", workers, len(chunks), chunk_size)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_transform_batch, chunk): i for i, chunk in enumerate(chunks)}
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        chunk_docs = future.result()
                        _process_and_stream(chunk_docs, f_norms, f_comps, f_tu)
                        logger.info(
                            "Pass 1: chunk %d/%d xong — %d văn bản trong chunk (tổng: %d Component, %d TextUnit đã ghi)",
                            chunk_idx + 1, len(chunks), len(chunk_docs), n_comp_total, n_tu_total,
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