"""
test.py — Debug chi tiết: tại sao action_extractor không khớp Component B?

Chạy 3 lớp kiểm tra:
  Layer 1  Structure   — structure_parser thấy được bao nhiêu Component?
  Layer 2  Index keys  — component_index của doc B có những key nào?
  Layer 3  Extraction  — action_extractor regex lấy được citation_path gì từ text doc A?
                         Có khớp với key nào trong component_index không?

Chạy:
    python test.py
    python test.py --id-a 8611 --id-b 11233
    python test.py --id-a 151936 --id-b 169557 --show-html
    python test.py --save-debug          # lưu debug/{id}_raw.html + _markdown.txt + _components.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, ".")

from transform.action_extractor import REGEX_PATTERNS, _normalize_citation_path
from transform.html_to_markdown import convert
from transform.pipeline import _build_component_index_entries
from transform.structure_parser import parse_structure
from transform.text_accumulator import build_accumulated_text, build_ancestor_chain
from schema.nodes import Norm
from datetime import datetime, timezone

DEFAULT_ID_A = 8611
DEFAULT_ID_B = 11233
CONTENT_PATH  = "data/raw/content.parquet"
META_PATH     = "data/raw/metadata.parquet"


# ─── helpers ─────────────────────────────────────────────────────────────────

def _load_by_ids(path: str, id_col: str, doc_ids: list, columns: list | None = None) -> dict:
    """Đọc parquet theo batch, lấy đúng các id cần — không load cả file."""
    pf = pq.ParquetFile(path)
    target = {int(i) for i in doc_ids}
    found: dict[int, dict] = {}
    kw = {"columns": columns} if columns else {}
    for batch in pf.iter_batches(batch_size=5000, **kw):
        for row in batch.to_pylist():
            doc_id = int(row[id_col])
            if doc_id in target:
                found[doc_id] = row
            if len(found) == len(target):
                return found
    return found


def _build_fake_norm(doc_id: int, meta_row: dict | None) -> Norm:
    return Norm(
        norm_id=str(doc_id),
        title=(meta_row or {}).get("title", ""),
        norm_number=(meta_row or {}).get("so_ky_hieu", ""),
        norm_type=(meta_row or {}).get("loai_van_ban", ""),
        updated_at=datetime.now(timezone.utc),
    )


def _build_component_data(doc_id: int, html: str, meta_row: dict | None):
    """Trả về (components, raw_text_map, leaf_texts, component_index_for_this_doc)."""
    if not html or not html.strip():
        return [], {}, {}, {}

    markdown = convert(html)
    if not markdown.strip():
        return [], {}, {}, {}

    result = parse_structure(str(doc_id), markdown)
    components = result.components
    if not components:
        return [], {}, {}, {}

    # Build accumulated_text cho leaf components (giống pipeline.py)
    norm = _build_fake_norm(doc_id, meta_row)
    comps_by_id = {c.comp_id: c for c in components}
    parent_ids = {c.parent_comp_id for c in components if c.parent_comp_id}
    leaves = [c for c in components if c.comp_id not in parent_ids]

    leaf_texts: dict[str, str] = {}
    for leaf in leaves:
        raw = result.raw_text.get(leaf.comp_id, "")
        if not raw.strip():
            continue
        chain = build_ancestor_chain(leaf, comps_by_id)
        leaf_texts[leaf.comp_id] = build_accumulated_text(norm, chain, raw.strip())

    # Build component_index với đầy đủ citation path (không chỉ citation đơn lẻ)
    index = _build_component_index_entries(str(doc_id), components)

    return components, result.raw_text, leaf_texts, index


# ─── save to debug/ ──────────────────────────────────────────────────────────

def save_debug(doc_id: int, html: str, components: list, raw_text: dict, leaf_texts: dict, index: dict) -> None:
    """Lưu toàn bộ dữ liệu của 1 doc ra debug/{id}_*.txt để đọc offline."""
    out = Path("debug")
    out.mkdir(exist_ok=True)

    # 1. HTML gốc
    (out / f"{doc_id}_raw.html").write_text(html or "(empty)", encoding="utf-8")

    # 2. Markdown sau convert
    markdown = convert(html) if html else ""
    (out / f"{doc_id}_markdown.txt").write_text(markdown or "(empty)", encoding="utf-8")

    # 3. Components + raw_text + leaf accumulated_text
    lines: list[str] = []
    parent_ids = {c.parent_comp_id for c in components if c.parent_comp_id}
    for comp in components:
        is_leaf = comp.comp_id not in parent_ids
        marker = "[LEAF]" if is_leaf else "[NODE]"
        lines.append(f"{marker}  {comp.comp_id}  [{comp.level.value}]  citation={comp.citation!r}  title={comp.title_text!r}")
        if is_leaf:
            raw = (raw_text.get(comp.comp_id) or "").strip()
            if raw:
                lines.append(f"  RAW:\n{_indent(raw)}")
            acc = (leaf_texts.get(comp.comp_id) or "").strip()
            if acc:
                lines.append(f"  ACCUMULATED:\n{_indent(acc)}")
        lines.append("")

    lines.append("─── component_index keys ───")
    for (nid, path), comp_id in sorted(index.items(), key=lambda x: x[0][1]):
        lines.append(f"  ({nid!r}, {path!r})  →  {comp_id}")

    (out / f"{doc_id}_components.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Saved: debug/{doc_id}_raw.html, debug/{doc_id}_markdown.txt, debug/{doc_id}_components.txt")


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln for ln in text.splitlines())


# ─── layer 1: structure ───────────────────────────────────────────────────────

def layer1_structure(doc_id: int, components: list) -> None:
    print(f"\n┌─ [Layer 1] Structure doc {doc_id}: {len(components)} Component")
    if not components:
        print("│  ❌ 0 Component — không có entry nào trong component_index")
        return
    parent_ids = {c.parent_comp_id for c in components if c.parent_comp_id}
    for comp in components:
        is_leaf = comp.comp_id not in parent_ids
        marker = "◆" if is_leaf else "○"
        indent = "  " * {"PHAN": 0, "CHUONG": 1, "MUC": 2, "TIEU_MUC": 2,
                          "DIEU": 2, "KHOAN": 3, "DIEM": 4}.get(comp.level.value, 0)
        title = (comp.title_text or "")[:60]
        print(f"│  {marker} {indent}{comp.comp_id}  [{comp.level.value}]  {comp.citation!r}  {title!r}")
    print("│")


# ─── layer 2: component_index keys ───────────────────────────────────────────

def layer2_index(doc_id: int, index: dict) -> None:
    print(f"├─ [Layer 2] component_index keys cho doc {doc_id}: {len(index)} key")
    if not index:
        print("│  ❌ 0 key — action_extractor sẽ không bao giờ khớp được")
        return
    for (nid, path), comp_id in sorted(index.items(), key=lambda x: x[0][1]):
        print(f"│  ({nid!r}, {path!r})  →  {comp_id}")
    print("│")


# ─── layer 3: regex extraction từ doc A ──────────────────────────────────────

def _leaf_body(text: str) -> str:
    """Trả về phần body của accumulated text (bỏ header [Chương X > Điều Y])."""
    if text.startswith("["):
        nl = text.find("\n")
        return text[nl + 1:].strip() if nl != -1 else text
    return text.strip()


def layer3_extraction(doc_id_a: int, leaf_texts: dict, doc_id_b: int, index_b: dict) -> None:
    print(f"├─ [Layer 3] Regex extraction từ doc A={doc_id_a} → tìm Component B trong doc {doc_id_b}")
    if not leaf_texts:
        print("│  ❌ Không có leaf text trong doc A — không có gì để regex")
        return

    hits = 0
    misses = 0
    no_regex: list[tuple[str, str]] = []  # (comp_a_id, body) cho leaf regex không bắt được

    for comp_a_id, text in leaf_texts.items():
        if not text.strip():
            continue
        extracted: str | None = None
        pattern_name: str | None = None
        for pat in REGEX_PATTERNS:
            m = pat.search(text)
            if m:
                extracted = _normalize_citation_path(m)
                pattern_name = pat.pattern[:40]
                break

        if extracted is None:
            no_regex.append((comp_a_id, _leaf_body(text)))
            continue

        lookup_key = (str(doc_id_b), extracted)
        matched = lookup_key in index_b

        status = "✅ MATCH" if matched else "❌ NO MATCH"
        print(f"│  {status}")
        print(f"│    comp_a:   {comp_a_id}")
        print(f"│    regex:    {pattern_name!r}...")
        print(f"│    extracted: {extracted!r}")
        if not matched:
            print(f"│    lookup:   ({str(doc_id_b)!r}, {extracted!r})  ← không có trong index_b")
            candidates = [k for k in index_b if extracted.split(" > ")[-1] in k[1]]
            if candidates:
                print(f"│    gần nhất: {candidates[:3]}")
        print(f"│    text (200c): {text[:200]!r}")
        print("│")
        if matched:
            hits += 1
        else:
            misses += 1

    if hits == 0 and misses == 0:
        print("│  ℹ️  Không có leaf nào trích được citation (regex không khớp câu nào)")
    print(f"├─ Kết quả: {hits} MATCH, {misses} NO MATCH")
    print(f"│")

    # Scan compact: hiện text thô của MỌI leaf (kể cả leaf regex không bắt được)
    # để user kiểm tra tay xem leaf nào thực sự đề cập đến doc B.
    print(f"├─ [Scan] Tất cả {len(leaf_texts)} leaf doc A — text thô để kiểm tra tay:")
    for comp_a_id, text in leaf_texts.items():
        body = _leaf_body(text)
        snippet = body[:120].replace("\n", " ")
        for pat in REGEX_PATTERNS:
            if pat.search(text):
                tag = "🔍"  # regex đã xử lý ở trên
                break
        else:
            tag = "  "
        print(f"│  {tag} {comp_a_id}  \"{snippet}\"")
    print(f"└─ (tổng {len(leaf_texts)} leaf)")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-a", type=int, default=DEFAULT_ID_A, help="doc_id nguồn (văn bản sửa đổi)")
    parser.add_argument("--id-b", type=int, default=DEFAULT_ID_B, help="doc_id đích (văn bản bị sửa)")
    parser.add_argument("--content", default=CONTENT_PATH)
    parser.add_argument("--meta", default=META_PATH)
    parser.add_argument("--show-html", action="store_true", help="In 500 ký tự HTML đầu của mỗi doc")
    parser.add_argument("--save-debug", action="store_true", help="Lưu HTML, markdown, components ra debug/")
    args, _ = parser.parse_known_args()

    id_a, id_b = args.id_a, args.id_b
    print(f"\nDebug: doc A={id_a} (nguồn/sửa đổi)  →  doc B={id_b} (đích/bị sửa)")
    print(f"Content: {args.content}")

    # Load HTML
    content_map = _load_by_ids(args.content, "id", [id_a, id_b], ["id", "content_html"])
    meta_map    = _load_by_ids(args.meta,    "id", [id_a, id_b])

    for did in [id_a, id_b]:
        if did not in content_map:
            print(f"⚠️  Không tìm thấy doc {did} trong {args.content}")

    html_a = (content_map.get(id_a) or {}).get("content_html", "")
    html_b = (content_map.get(id_b) or {}).get("content_html", "")

    if args.show_html:
        print(f"--- HTML doc A (500c) ---\n{repr(html_a[:500])}\n")
        print(f"--- HTML doc B (500c) ---\n{repr(html_b[:500])}\n")

    comps_a, raw_a, leaf_a, index_a = _build_component_data(id_a, html_a, meta_map.get(id_a))
    comps_b, raw_b, leaf_b, index_b = _build_component_data(id_b, html_b, meta_map.get(id_b))

    if args.save_debug:
        print("\nLưu debug files...")
        save_debug(id_a, html_a, comps_a, raw_a, leaf_a, index_a)
        save_debug(id_b, html_b, comps_b, raw_b, leaf_b, index_b)

    print(f"\n{'═'*70}")
    print(f"DOC A (nguồn/sửa đổi): {id_a}  —  {len(comps_a)} Component, {len(leaf_a)} leaf")
    print(f"{'═'*70}")
    layer1_structure(id_a, comps_a)

    print(f"\n{'═'*70}")
    print(f"DOC B (đích/bị sửa):   {id_b}  —  {len(comps_b)} Component, {len(leaf_b)} leaf")
    print(f"{'═'*70}")
    layer1_structure(id_b, comps_b)
    layer2_index(id_b, index_b)

    print(f"\n{'═'*70}")
    print("CROSS-CHECK: regex doc A  →  index doc B")
    print(f"{'═'*70}")
    layer3_extraction(id_a, leaf_a, id_b, index_b)


if __name__ == "__main__":
    main()
