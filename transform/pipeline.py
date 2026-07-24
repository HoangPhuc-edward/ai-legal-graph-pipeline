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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc

from config import LEVEL_RANK, TRANSFORMED_DIR
from schema.edges import NormRelation
from schema.enums import ComponentLevel
from schema.nodes import Action, Component, Norm, TextUnit
from transform import relation_classifier
from transform.html_to_markdown import convert as html_to_markdown
from transform.structure_parser import parse_structure
from transform.text_accumulator import build_accumulated_text, build_ancestor_chain
from transform.validate_structure import StructuralWarning, validate_structure

logger = logging.getLogger(__name__)


# ─── Dedup & Validation ────────────────────────────────────────────────────────

def _raw_text_score(text: str) -> float:
    """Điểm chất lượng raw_text — dùng để chọn giữa 2 Component trùng citation.

    Ưu tiên (giảm dần):
    1. Text không bắt đầu bằng "Bảng [" (table metadata noise) — hệ số 1.0
    2. Text dài hơn (nhiều nội dung hơn)
    Text bắt đầu "Bảng [" bị phạt ×0.1 — chỉ giữ khi không có lựa chọn nào khác.
    """
    if not text:
        return 0.0
    # "Bảng [" = preamble bị nuốt vào table header → luôn thua plain text dù dài hơn.
    # Hệ số 0.001 đảm bảo plain text 1 char (score=1) > table text 999 char (score=0.999).
    return len(text) * (0.001 if text.startswith("Bảng [") else 1.0)


def _dedup_components(
    components: list[Component],
    raw_text: Optional[dict[str, str]] = None,
) -> list[Component]:
    """Loại bỏ Component trùng (parent_comp_id, citation) — best-quality-wins.

    Khi raw_text được cung cấp: giữ Component có raw_text chất lượng cao nhất
    (không phải table metadata, và dài hơn). Khi không có raw_text: first-wins.

    Cascade: nếu một Component bị loại, tất cả con/cháu cũng bị loại theo.
    Điều này xử lý HTML triplication mà không xoá Phụ lục hợp lệ (Phụ lục
    đã có ComponentLevel.PHU_LUC làm parent riêng).

    Giả định: components sắp xếp theo order_index tăng dần.
    """
    # Pass 1: tìm winner cho mỗi (parent, citation) key
    winner: dict[tuple, str] = {}  # key → comp_id của winner
    winner_score: dict[tuple, float] = {}

    for c in components:
        key = (c.parent_comp_id, c.citation)
        if raw_text is not None:
            score = _raw_text_score(raw_text.get(c.comp_id, ""))
        else:
            score = 1.0 if key not in winner else -1.0  # first-wins khi không có text

        if key not in winner or score > winner_score[key]:
            winner[key] = c.comp_id
            winner_score[key] = score

    winner_ids = set(winner.values())

    # Pass 2: cascade — loại comp trùng + tất cả con/cháu của chúng
    dropped: set[str] = set()
    result: list[Component] = []

    for c in components:
        if c.parent_comp_id in dropped:
            dropped.add(c.comp_id)
            continue
        key = (c.parent_comp_id, c.citation)
        if c.comp_id != winner.get(key):
            dropped.add(c.comp_id)
        else:
            result.append(c)

    if dropped:
        logger.debug("_dedup_components: loại %d Component trùng", len(dropped))
    return result


def _remove_triplication_ghosts(components: list[Component]) -> list[Component]:
    """Pass 3 sau dedup: loại ghost DIEUs từ HTML triplication còn sót.

    Symptom: parent P có dãy DIEU [44, 45, ..., 52, 1, 2, 3, 4] — số giảm đột ngột.
    Nguyên nhân: copy 2/3 của HTML xuất hiện khi stack đang có P (thường là Chương cuối),
    làm preamble DIEUs của copy 2 bị gắn vào P thay vì ROOT. Dedup không bắt được
    vì (P_id, "Điều 1") ≠ (None, "Điều 1").

    Heuristic: tail DIEU (sau điểm reset số) có citation đã tồn tại dưới parent NÔNG hơn
    (gần ROOT hơn trong hierarchy) → ghost → drop + cascade children.

    Không xử lý trường hợp PHU_LUC có Điều riêng: PHU_LUC rank=0 (cùng level với Phan),
    nên parent_depth(DIEU trong PHU_LUC) = 0 và DIEU của thân văn bản (dưới ROOT) = -1.
    Nghĩa là DIEU trong PHU_LUC SAU hơn so với DIEU ở ROOT → thuật toán sẽ drop chúng
    nếu có reset... trừ khi bản thân PHU_LUC cũng tạo ra reset (vì PHU_LUC là parent riêng,
    DIEU của PHU_LUC và DIEU của thân không share cùng parent group → không có reset trong
    group đó → không bị xử lý ở đây). An toàn.
    """
    import re as _re
    _dieu_num = _re.compile(r"Điều\s+(\d+)", _re.IGNORECASE)

    comp_by_id = {c.comp_id: c for c in components}

    def _parent_depth(c: Component) -> int:
        """Depth của parent trong hierarchy (-1=ROOT, 0=PhuLuc/Phan, 1=Chuong, ...)."""
        if c.parent_comp_id is None:
            return -1
        p = comp_by_id.get(c.parent_comp_id)
        return LEVEL_RANK.get(p.level.value, 99) if p else 99

    # Index: citation → [(depth, comp_id)] chỉ cho DIEU
    dieu_by_citation: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for c in components:
        if c.level == ComponentLevel.DIEU:
            dieu_by_citation[c.citation].append((_parent_depth(c), c.comp_id))

    # Nhóm siblings theo parent
    siblings: dict[Optional[str], list[Component]] = defaultdict(list)
    for c in components:
        siblings[c.parent_comp_id].append(c)

    to_drop: set[str] = set()

    for parent_id, sibs in siblings.items():
        dieu_sibs = [c for c in sibs if c.level == ComponentLevel.DIEU]
        if len(dieu_sibs) < 2:
            continue

        # Tìm điểm reset (số giảm lần đầu tiên)
        prev_num = None
        reset_idx: Optional[int] = None
        for i, c in enumerate(dieu_sibs):
            m = _dieu_num.search(c.citation)
            num = int(m.group(1)) if m else -1
            if prev_num is not None and num >= 0 and prev_num >= 0 and num < prev_num:
                reset_idx = i
                break
            if num >= 0:
                prev_num = num

        if reset_idx is None:
            continue

        # Tail = từ điểm reset — check từng ghost
        for ghost_c in dieu_sibs[reset_idx:]:
            this_depth = _parent_depth(ghost_c)
            has_shallower = any(
                d < this_depth
                for d, cid in dieu_by_citation.get(ghost_c.citation, [])
                if cid != ghost_c.comp_id
            )
            if has_shallower:
                to_drop.add(ghost_c.comp_id)

    if not to_drop:
        return components

    # Cascade: drop ghost + tất cả con/cháu
    dropped: set[str] = set(to_drop)
    result: list[Component] = []
    for c in components:
        if c.parent_comp_id in dropped:
            dropped.add(c.comp_id)
        elif c.comp_id not in dropped:
            result.append(c)

    logger.debug("_remove_triplication_ghosts: dropped %d ghost components", len(dropped))
    return result


@dataclass
class StructureIssue:
    norm_id: str
    issue_type: str   # "dup_citation" | "broken_parent" | "orphan_text"
    severity: str     # "warn" | "error"
    message: str
    details: dict


def validate_parse_result(norm_id: str, result) -> list[StructureIssue]:
    """Kiểm tra tính nhất quán của ParseResult SAU khi dedup.

    Sau dedup không nên còn lỗi — nếu vẫn còn là dấu hiệu parse sai cấu trúc
    thực sự (không phải HTML triplication đơn giản). Caller có thể dùng danh
    sách này để quyết định có cần LLM fallback hay không.
    """
    issues: list[StructureIssue] = []
    components_by_id = {c.comp_id: c for c in result.components}

    # 1. Duplicate (parent_comp_id, citation) còn sót sau dedup
    citation_groups: dict[tuple, list[str]] = defaultdict(list)
    for c in result.components:
        citation_groups[(c.parent_comp_id, c.citation)].append(c.comp_id)
    for (parent_id, citation), comp_ids in citation_groups.items():
        if len(comp_ids) > 1:
            issues.append(StructureIssue(
                norm_id=norm_id,
                issue_type="dup_citation",
                severity="warn",
                message=(
                    f"Citation {citation!r} vẫn trùng {len(comp_ids)}× "
                    f"dưới parent={parent_id} — nội dung khác nhau, không dedup được"
                ),
                details={"parent_id": parent_id, "citation": citation, "comp_ids": comp_ids},
            ))

    # 2. Parent reference trỏ đến comp_id không tồn tại
    for c in result.components:
        if c.parent_comp_id and c.parent_comp_id not in components_by_id:
            issues.append(StructureIssue(
                norm_id=norm_id,
                issue_type="broken_parent",
                severity="error",
                message=f"Component {c.comp_id} trỏ đến parent {c.parent_comp_id} không tồn tại",
                details={"comp_id": c.comp_id, "missing_parent": c.parent_comp_id},
            ))

    # 3. raw_text gắn vào comp_id không có trong components_by_id
    for comp_id in result.raw_text:
        if comp_id not in components_by_id:
            issues.append(StructureIssue(
                norm_id=norm_id,
                issue_type="orphan_text",
                severity="error",
                message=f"raw_text gắn vào comp_id {comp_id!r} không tồn tại trong components",
                details={"comp_id": comp_id},
            ))

    # 4. Non-monotonic DIEU sequence trong cùng parent (dấu hiệu HTML triplication ghost)
    # Điều số giảm trong cùng group → likely copy N của tài liệu bị gắn nhầm parent.
    import re as _re
    _dieu_num_pat = _re.compile(r"Điều\s+(\d+)", _re.IGNORECASE)
    siblings_by_parent: dict[Optional[str], list] = defaultdict(list)
    for c in result.components:
        siblings_by_parent[c.parent_comp_id].append(c)

    for parent_id, sibs in siblings_by_parent.items():
        dieu_sibs = [c for c in sibs if c.level == ComponentLevel.DIEU]
        if len(dieu_sibs) < 2:
            continue
        prev_num = None
        for c in dieu_sibs:
            m = _dieu_num_pat.search(c.citation)
            if not m:
                continue
            num = int(m.group(1))
            if prev_num is not None and num < prev_num:
                parent_comp = components_by_id.get(parent_id or "")
                parent_cit = parent_comp.citation if parent_comp else "ROOT"
                issues.append(StructureIssue(
                    norm_id=norm_id,
                    issue_type="ghost_citation",
                    severity="warn",
                    message=(
                        f"Điều số GIẢM trong {parent_cit}: {prev_num} → {num} "
                        f"(likely HTML triplication ghost dưới sai parent)"
                    ),
                    details={"parent_id": parent_id, "prev_num": prev_num, "curr_num": num},
                ))
                break  # 1 issue/parent group là đủ
            prev_num = num

    return issues


def _log_issues(norm_id: str, issues: list[StructureIssue]) -> None:
    """Log issues và đánh dấu cases cần LLM review."""
    if not issues:
        return
    dup_count = sum(1 for i in issues if i.issue_type == "dup_citation")
    error_count = sum(1 for i in issues if i.severity == "error")
    logger.warning(
        "Norm %s — %d issue(s): %d dup_citation, %d error",
        norm_id, len(issues), dup_count, error_count,
    )
    for issue in issues:
        fn = logger.error if issue.severity == "error" else logger.warning
        fn("  [%s/%s] %s", norm_id, issue.issue_type, issue.message)
    if dup_count > 0:
        # TODO: LLM fallback — gọi LLM để xác định Component nào là canonical
        # khi 2 Component có cùng (parent, citation) nhưng text khác nhau.
        # Đây là trường hợp phức tạp mà dedup không thể tự giải quyết.
        logger.warning(
            "  [%s] LLM fallback CHƯA IMPLEMENT — %d dup_citation cần review thủ công",
            norm_id, dup_count,
        )


def _log_structural_warnings(norm_id: str, warnings: list[StructuralWarning]) -> None:
    """Log kết quả validate_structure — fix/warn/bug theo mức độ."""
    if not warnings:
        return
    fix_count = sum(1 for w in warnings if w.severity == "fix")
    warn_count = sum(1 for w in warnings if w.severity == "warn")
    bug_count = sum(1 for w in warnings if w.severity == "bug")
    logger.info(
        "Norm %s — validate_structure: %d fix, %d warn, %d bug",
        norm_id, fix_count, warn_count, bug_count,
    )
    for w in warnings:
        if w.severity == "bug":
            logger.error("  [%s/%s] %s", norm_id, w.rule, w.message)
        elif w.severity == "fix":
            logger.info("  [%s/%s] %s", norm_id, w.rule, w.message)
        else:
            logger.warning("  [%s/%s] %s", norm_id, w.rule, w.message)


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

    # Pass 1: Dedup (parent, citation) trùng — HTML triplication phổ biến.
    # Truyền raw_text để chọn Component có nội dung tốt hơn (không phải table noise).
    parse_result.components = _dedup_components(parse_result.components, parse_result.raw_text)
    # Pass 2: Loại ghost DIEUs từ triplication còn sót sau dedup (khác parent).
    parse_result.components = _remove_triplication_ghosts(parse_result.components)
    kept_ids = {c.comp_id for c in parse_result.components}
    parse_result.raw_text = {k: v for k, v in parse_result.raw_text.items() if k in kept_ids}

    # Validate sau dedup — log warning/error nếu vẫn còn bất thường
    issues = validate_parse_result(norm.norm_id, parse_result)
    _log_issues(norm.norm_id, issues)

    # Kiểm tra và sửa cấu trúc (B1-E1): D2 tự sửa, các rule khác chỉ cảnh báo
    vs_result = validate_structure(norm.norm_id, parse_result.components, parse_result.raw_text)
    parse_result.components = vs_result.components
    parse_result.raw_text = vs_result.raw_text
    _log_structural_warnings(norm.norm_id, vs_result.warnings)

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
            # processed_ids dedup: content source có exact duplicate rows → bỏ qua lần 2+
            for batch in content_filtered.to_batches(max_chunksize=CONTENT_BATCH):
                batch_rows = batch.to_pylist()
                batch_pairs = []
                for r in batch_rows:
                    doc_id = str(r["id"])
                    if doc_id in processed_ids:
                        continue
                    processed_ids.add(doc_id)
                    if doc_id in meta_by_id:
                        batch_pairs.append((meta_by_id[doc_id], r["content_html"] or ""))
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