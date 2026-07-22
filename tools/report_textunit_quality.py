"""Báo cáo chất lượng TextUnit từ Neo4j.

Chạy:
    python tools/report_textunit_quality.py
    python tools/report_textunit_quality.py --sample 500
    python tools/report_textunit_quality.py --sample 1000 --out quality_report.txt
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Đảm bảo import từ root project (chạy từ bất kỳ thư mục nào)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from load.neo4j_client import Neo4jClient

# ── Regex ──────────────────────────────────────────────────────────────────────
_HEADER_RE = re.compile(r"^\[([\s\S]*?)\](?=\n|$)")  # non-greedy: dừng ở ]\n đầu tiên → đúng cho header nhiều dòng, bỏ qua ] trong body
_SEPARATOR_IN_HEADER = re.compile(r"[>→]")          # có phân cấp trong header
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # control ngoài \n \t
_PROSE_TABLE_RE = re.compile(r"^Bảng \[", re.MULTILINE)   # prose table header
_PROSE_EMPTY_RE = re.compile(r"^Bảng \[\]:", re.MULTILINE) # prose table rỗng (header trống)
_PIPE_TABLE_RE = re.compile(r"^\|.+\|", re.MULTILINE)      # pipe table còn sót
_PIPE_SEP_RE = re.compile(r"^\|\s*[-:]{2,}", re.MULTILINE) # | --- | separator row


# ── Cypher queries ──────────────────────────────────────────────────────────────

_QUERY_ALL = """
MATCH (c:Component)-[:HAS_TEXTUNIT]->(t:TextUnit)
WHERE t.type <> 'cache_action'
RETURN t.unit_id   AS unit_id,
       t.accumulated_text AS text,
       t.type      AS type,
       c.level     AS level,
       c.citation  AS citation
"""

_QUERY_SAMPLE = """
MATCH (c:Component)-[:HAS_TEXTUNIT]->(t:TextUnit)
WHERE t.type <> 'cache_action'
WITH t, c, rand() AS r
ORDER BY r
LIMIT $n
RETURN t.unit_id   AS unit_id,
       t.accumulated_text AS text,
       t.type      AS type,
       c.level     AS level,
       c.citation  AS citation
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _body(text: str | None) -> str:
    """Phần nội dung sau header [...]."""
    if not text:
        return ""
    m = _HEADER_RE.match(text)
    if not m:
        return text
    rest = text[m.end():]
    return rest.lstrip("\n")


def _classify_format(text: str | None) -> str:
    """Phân loại format accumulated_text."""
    if text is None:
        return "null_text"
    if not text:
        return "no_header"
    m = _HEADER_RE.match(text)
    if not m:
        return "no_header"
    header_content = m.group(1)
    if not _SEPARATOR_IN_HEADER.search(header_content):
        return "header_no_separator"   # văn bản 1 cấp — informational, không phải lỗi
    return "ok"


# ── Report sections ────────────────────────────────────────────────────────────

def _report_fallback(rows: list[dict], out) -> bool:
    total = len(rows)
    full_count = sum(1 for r in rows if "_FULL_" in (r["citation"] or ""))
    structured = total - full_count
    pct_full = full_count / total * 100 if total else 0.0

    print("═" * 60, file=out)
    print("① FALLBACK CITATION (_FULL_)", file=out)
    print("═" * 60, file=out)
    print(f"  Tổng Component có TextUnit : {total:,}", file=out)
    print(f"  Tách đúng cấu trúc         : {structured:,}  ({100 - pct_full:.1f}%)", file=out)
    print(f"  Fallback _FULL_            : {full_count:,}  ({pct_full:.1f}%)", file=out)

    issue = pct_full > 5.0
    status = "⚠  CẦN XEM LẠI" if issue else "✓  OK"
    print(f"  → {status}", file=out)
    print(file=out)
    return issue


def _report_format(rows: list[dict], out) -> bool:
    total = len(rows)
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {
        "null_text": [],
        "no_header": [],
        "header_no_separator": [],
    }

    for r in rows:
        cls = _classify_format(r["text"])
        counts[cls] += 1
        if cls not in ("ok", "header_no_separator") and len(examples.get(cls, [])) < 3:
            examples.setdefault(cls, []).append(r["unit_id"])

    labels = {
        "ok":                 "Đúng format (đa cấp)",
        "header_no_separator":"Văn bản 1 cấp (không có >)",
        "no_header":          "Không có header [...] ⚠",
        "null_text":          "accumulated_text = null ⚠",
    }

    print("═" * 60, file=out)
    print("② FORMAT accumulated_text", file=out)
    print("═" * 60, file=out)
    for key, label in labels.items():
        n = counts[key]
        pct = n / total * 100 if total else 0.0
        print(f"  {label:<40} {n:>7,}  ({pct:5.1f}%)", file=out)
        if examples.get(key):
            print(f"    Ví dụ: {', '.join(examples[key])}", file=out)
    print(file=out)

    # header_no_separator là valid data (1 cấp) — không flag
    issue = counts["no_header"] + counts["null_text"] > 0
    status = "⚠  CẦN XEM LẠI" if issue else "✓  OK"
    print(f"  → {status}", file=out)
    print(file=out)
    return issue


def _report_text_quality(rows: list[dict], out) -> bool:
    replacement_count = 0
    ctrl_count = 0
    len_dist: Counter[str] = Counter()
    replacement_examples: list[str] = []
    ctrl_examples: list[str] = []

    for r in rows:
        body = _body(r["text"] or "")
        n = len(body)

        if n == 0:
            len_dist["rỗng"] += 1
        elif n < 20:
            len_dist["<20"] += 1
        elif n < 100:
            len_dist["20–100"] += 1
        elif n < 1000:
            len_dist["100–1000"] += 1
        else:
            len_dist[">1000"] += 1

        if "�" in body:
            replacement_count += 1
            if len(replacement_examples) < 3:
                replacement_examples.append(r["unit_id"])

        if _CTRL_CHARS.search(body):
            ctrl_count += 1
            if len(ctrl_examples) < 3:
                ctrl_examples.append(r["unit_id"])

    total = len(rows)

    print("═" * 60, file=out)
    print("③ CHẤT LƯỢNG TEXT (phần nội dung sau header)", file=out)
    print("═" * 60, file=out)

    print("  Ký tự lỗi:", file=out)
    _pct = lambda n: f"{n:,}  ({n / total * 100:.1f}%)" if total else str(n)
    print(f"    \\ufffd (decode sai)   : {_pct(replacement_count)}", file=out)
    if replacement_examples:
        print(f"      Ví dụ: {', '.join(replacement_examples)}", file=out)
    print(f"    Control chars lạ   : {_pct(ctrl_count)}", file=out)
    if ctrl_examples:
        print(f"      Ví dụ: {', '.join(ctrl_examples)}", file=out)

    print("  Phân bố độ dài:", file=out)
    for bucket in ["rỗng", "<20", "20–100", "100–1000", ">1000"]:
        n = len_dist[bucket]
        pct = n / total * 100 if total else 0.0
        bar = "█" * int(pct / 2)
        print(f"    {bucket:<12} {n:>7,}  ({pct:5.1f}%)  {bar}", file=out)
    print(file=out)

    issue = replacement_count > 0 or ctrl_count > 0
    status = "⚠  CẦN XEM LẠI" if issue else "✓  OK"
    print(f"  → {status}", file=out)
    print(file=out)
    return issue


def _report_tables(rows: list[dict], out) -> bool:
    total = len(rows)
    has_prose = 0          # TextUnit có ít nhất 1 "Bảng [...]:"
    has_empty_header = 0   # TextUnit có "Bảng []:" (header trống)
    has_pipe = 0           # TextUnit còn pipe table chưa convert
    table_count: Counter[int] = Counter()   # phân bố số bảng/TextUnit
    empty_examples: list[str] = []
    pipe_examples: list[str] = []

    for r in rows:
        body = _body(r["text"] or "")

        prose_hits = _PROSE_TABLE_RE.findall(body)
        n_tables = len(prose_hits)
        table_count[n_tables] += 1
        if n_tables > 0:
            has_prose += 1

        if _PROSE_EMPTY_RE.search(body):
            has_empty_header += 1
            if len(empty_examples) < 3:
                empty_examples.append(r["unit_id"])

        # Pipe table: phải có cả dòng data VÀ dòng separator |---|
        if _PIPE_TABLE_RE.search(body) and _PIPE_SEP_RE.search(body):
            has_pipe += 1
            if len(pipe_examples) < 3:
                pipe_examples.append(r["unit_id"])

    pct = lambda n: f"{n:,}  ({n / total * 100:.1f}%)" if total else str(n)

    print("═" * 60, file=out)
    print("④ CHẤT LƯỢNG BẢNG BIỂU", file=out)
    print("═" * 60, file=out)

    print(f"  TextUnit có bảng prose      : {pct(has_prose)}", file=out)
    print(f"  Pipe table còn sót          : {pct(has_pipe)}", file=out)
    if pipe_examples:
        print(f"    Ví dụ: {', '.join(pipe_examples)}", file=out)
    print(f"  Bảng rỗng (\"Bảng []\")       : {pct(has_empty_header)}", file=out)
    if empty_examples:
        print(f"    Ví dụ: {', '.join(empty_examples)}", file=out)

    print("  Phân bố số bảng/TextUnit:", file=out)
    for bucket in [0, 1, 2, 3, 4, 5]:
        n = table_count[bucket]
        pct_b = n / total * 100 if total else 0.0
        bar = "█" * int(pct_b / 2)
        label = f"{bucket} bảng" if bucket < 5 else "≥5 bảng"
        # Bucket 5 accumulates 5+
        if bucket == 5:
            n = sum(v for k, v in table_count.items() if k >= 5)
            pct_b = n / total * 100 if total else 0.0
            bar = "█" * int(pct_b / 2)
        print(f"    {label:<10} {n:>7,}  ({pct_b:5.1f}%)  {bar}", file=out)
    print(file=out)

    issue = has_pipe > 0 or has_empty_header > 0
    status = "⚠  CẦN XEM LẠI" if issue else "✓  OK"
    print(f"  → {status}", file=out)
    print(file=out)
    return issue


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Báo cáo chất lượng TextUnit từ Neo4j")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Lấy ngẫu nhiên N TextUnit (mặc định: toàn bộ)")
    parser.add_argument("--out", type=str, default=None, metavar="FILE",
                        help="Ghi report ra file (mặc định: stdout)")
    args = parser.parse_args()

    out_file = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout

    try:
        print("Kết nối Neo4j...", file=sys.stderr)
        with Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as client:
            if args.sample:
                print(f"Lấy mẫu {args.sample:,} TextUnit...", file=sys.stderr)
                rows = client.query(_QUERY_SAMPLE, n=args.sample)
            else:
                print("Lấy toàn bộ TextUnit...", file=sys.stderr)
                rows = client.query(_QUERY_ALL)

        if not rows:
            print("Không có TextUnit nào trong DB (type != cache_action).", file=out_file)
            return

        total = len(rows)
        label = f"mẫu {total:,}" if args.sample else f"toàn bộ {total:,}"
        print(f"\n{'═' * 60}", file=out_file)
        print(f"  BÁO CÁO CHẤT LƯỢNG TEXTUNIT  —  {label} bản", file=out_file)
        print(f"{'═' * 60}\n", file=out_file)

        issues = []
        if _report_fallback(rows, out_file):
            issues.append("① Fallback _FULL_ cao")
        if _report_format(rows, out_file):
            issues.append("② Format sai")
        if _report_text_quality(rows, out_file):
            issues.append("③ Ký tự lỗi")
        if _report_tables(rows, out_file):
            issues.append("④ Bảng biểu lỗi")

        print("═" * 60, file=out_file)
        print("TỔNG KẾT", file=out_file)
        print("═" * 60, file=out_file)
        if issues:
            print(f"  ⚠  Cần xem lại: {' | '.join(issues)}", file=out_file)
        else:
            print("  ✓  Tất cả đánh giá đạt — TextUnit chất lượng tốt.", file=out_file)
        print(file=out_file)

    finally:
        if args.out and out_file is not sys.stdout:
            out_file.close()
            print(f"Report đã ghi ra: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
