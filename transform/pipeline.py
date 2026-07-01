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

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa

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


def transform_relationships(
    relationships_table: pa.Table,
    docs_by_norm_id: dict[str, TransformedDoc],
    component_index: dict[tuple[str, str], str],
    use_llm: bool = True,
) -> tuple[list[NormRelation], list[tuple[Action, TextUnit, str, str]]]:
    """PASS 2 — xử lý relationships.parquet bằng component_index đã build ở Pass 1.
    Tầng B chỉ chạy cho các dòng mà doc_id (nguồn) đã có trong docs_by_norm_id."""

    def get_component_text_map(norm_id: str) -> dict[str, str]:
        doc = docs_by_norm_id.get(norm_id)
        return doc.component_text if doc else {}

    def lookup_norm_number(norm_id: str) -> str:
        doc = docs_by_norm_id.get(norm_id)
        return doc.norm.norm_number if doc else norm_id

    relations: list[NormRelation] = []
    actions: list[tuple[Action, TextUnit, str, str]] = []

    for row in relationships_table.to_pylist():
        doc_id = str(row["doc_id"])
        other_doc_id = str(row["other_doc_id"])
        label = row["relationship"]
        if doc_id not in docs_by_norm_id:
            continue
        for item in relation_classifier.process_relationship_row(
            doc_id=doc_id,
            other_doc_id=other_doc_id,
            relationship_label=label,
            component_index=component_index,
            get_component_text_map=get_component_text_map,
            lookup_norm_number=lookup_norm_number,
            use_llm=use_llm,
        ):
            if isinstance(item, NormRelation):
                relations.append(item)
            else:
                actions.append(item)

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

    workers=1 (mặc định, an toàn cho laptop 8GB): sequential, không spawn process.
    workers>1 (gợi ý Colab: 4): Pass 1 chạy parallel bằng ProcessPoolExecutor —
      mỗi worker xử lý 1 chunk văn bản độc lập, main process merge kết quả và
      build component_index sau khi TẤT CẢ worker xong (không parallel được vì
      cần toàn bộ Component trước khi index hoàn chỉnh).
    """
    metadata_rows = metadata_table.to_pylist()
    if sample is not None:
        metadata_rows = metadata_rows[:sample]

    content_by_id = {row["id"]: row["content_html"] for row in content_table.to_pylist()}

    # ═══════════ PASS 1 — structure parsing toàn corpus + build component_index ═══════════
    docs_by_norm_id: dict[str, TransformedDoc] = {}
    component_index: dict[tuple[str, str], str] = {}

    pairs = [
        (row, content_by_id.get(row["id"]) or content_by_id.get(str(row["id"])) or "")
        for row in metadata_rows
    ]

    if workers <= 1:
        all_docs = _transform_batch(pairs)
    else:
        chunk_size = max(1, len(pairs) // workers)
        chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        logger.info("Pass 1 — %d văn bản / %d worker / chunk ~%d", len(pairs), len(chunks), chunk_size)
        all_docs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_transform_batch, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    all_docs.extend(future.result())
                except Exception:
                    logger.exception("Worker chunk %d lỗi, bỏ qua.", chunk_idx)

    for doc in all_docs:
        docs_by_norm_id[doc.norm.norm_id] = doc
        component_index.update(_build_component_index_entries(doc.norm.norm_id, doc.components))

    # ═══════════ PASS 2 — action extraction xuyên văn bản, dùng component_index ═══════════
    relations, actions = transform_relationships(
        relationships_table, docs_by_norm_id, component_index, use_llm=use_llm
    )

    norms = [doc.norm for doc in docs_by_norm_id.values()]
    components = [c for doc in docs_by_norm_id.values() for c in doc.components]
    component_text_units = [tu for doc in docs_by_norm_id.values() for tu in doc.text_units]
    cache_text_units = [tu for _, tu, _, _ in actions]
    action_objs = [a for a, _, _, _ in actions]

    _write_jsonl(output_dir / "norms.jsonl", norms)
    _write_jsonl(output_dir / "components.jsonl", components)
    _write_jsonl(output_dir / "textunits.jsonl", component_text_units + cache_text_units)
    _write_jsonl(output_dir / "actions.jsonl", action_objs)
    _write_jsonl(output_dir / "relations.jsonl", relations)

    # component_textunit_map: comp_id -> unit_id (HAS_TEXTUNIT từ Component, type="noi_dung")
    component_textunit_map = {
        comp_id: unit_id
        for doc in docs_by_norm_id.values()
        for comp_id, unit_id in doc.component_text_unit.items()
    }
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
        len(norms),
        len(components),
        len(component_text_units) + len(cache_text_units),
        len(cache_text_units),
        len(action_objs),
        len(relations),
    )
