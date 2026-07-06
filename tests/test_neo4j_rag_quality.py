"""Kiểm tra chất lượng dữ liệu Neo4j sau khi load — đảm bảo đồ thị đúng
và đủ để RAG hoạt động. Không kiểm tra số lượng tuyệt đối.

Yêu cầu: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD đã set trong .env.
Nếu chưa kết nối được, toàn bộ test tự động skip.

TC-04 / TC-05 / TC-06 cần embed đã chạy và vector index đã tạo thủ công:
    CREATE VECTOR INDEX textunit_embedding_index IF NOT EXISTS
    FOR (t:TextUnit) ON (t.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};

Thứ tự ưu tiên: TC-01 → TC-02 → TC-04 → TC-06 (đủ xác nhận RAG chạy được).

Chạy:
    pytest tests/test_neo4j_rag_quality.py -v
    pytest tests/test_neo4j_rag_quality.py -v -m critical
"""
from __future__ import annotations

import pytest

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

_VECTOR_INDEX_NAME = "textunit_embedding_index"


# ─────────────────────────── kết nối & skip ────────────────────────────


def _try_connect() -> bool:
    if not NEO4J_URI:
        return False
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        d.verify_connectivity()
        d.close()
        return True
    except Exception:
        return False


_neo4j_ok = _try_connect()
pytestmark = pytest.mark.skipif(
    not _neo4j_ok,
    reason="NEO4J_URI chưa set hoặc không kết nối được — xem README.md mục Test",
)


@pytest.fixture(scope="module")
def driver():
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield d
    d.close()


def _q(driver, cypher: str, **params) -> list[dict]:
    with driver.session() as s:
        return s.run(cypher, **params).data()


def _vector_index_exists(driver) -> bool:
    try:
        rows = _q(driver, "SHOW VECTOR INDEXES")
        return any(r.get("name") == _VECTOR_INDEX_NAME for r in rows)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════
# NHÓM 1 — Cấu trúc đồ thị đúng chưa?
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.critical
def test_tc01_la_co_textunit(driver):
    """TC-01: ≥95% node lá (không có Component con) có TextUnit.

    Chỉ node lá là đơn vị nội dung thực sự — Chương/Mục/Điều có Khoản con
    không cần lưu nội dung riêng.
    """
    rows = _q(driver, """
        MATCH (c:Component)
        WHERE NOT (c)-[:CONTAINS]->(:Component)
        WITH count(c) AS tong_la
        MATCH (la:Component)-[:HAS_TEXTUNIT]->(tu:TextUnit)
        WHERE NOT (la)-[:CONTAINS]->(:Component)
        RETURN
          tong_la,
          count(DISTINCT la) AS la_co_textunit,
          round(100.0 * count(DISTINCT la) / tong_la, 1) AS phan_tram
    """)
    r = rows[0]
    if r["tong_la"] == 0:
        pytest.skip("Không có Component nào — bước load chưa chạy")
    assert r["phan_tram"] >= 95.0, (
        f"Chỉ {r['phan_tram']}% node lá có TextUnit (ngưỡng: ≥95%). "
        f"{r['la_co_textunit']}/{r['tong_la']} node lá. "
        "Kiểm tra text_accumulator hoặc load_component_textunits()."
    )


@pytest.mark.critical
def test_tc02_cay_khong_dut_gay(driver):
    """TC-02: 0 Component không truy ngược lên được Norm.

    Cây đứt gãy khiến RAG tìm được nội dung nhưng không biết thuộc văn bản nào.
    """
    rows = _q(driver, """
        MATCH (c:Component)
        WHERE NOT (:Norm)-[:CONTAINS*]->(c)
        RETURN count(c) AS component_mo_neo, collect(c.comp_id)[..5] AS vi_du
    """)
    r = rows[0]
    assert r["component_mo_neo"] == 0, (
        f"{r['component_mo_neo']} Component không có Norm cha: {r['vi_du']}. "
        "Kiểm tra thứ tự load — Component phải load sau Norm."
    )


@pytest.mark.critical
def test_tc03_action_du_hai_dau(driver):
    """TC-03: 0 Action thiếu Component nguồn (HAS_ACTION) hoặc đích (APPLY_TO).

    Action thiếu 1 đầu thì graph expansion trong RAG bị chết.
    """
    rows = _q(driver, """
        MATCH (act:Action)
        WITH act,
          COUNT { (:Component)-[:HAS_ACTION]->(act) } > 0 AS co_nguon,
          COUNT { (act)-[:APPLY_TO]->(:Component) } > 0 AS co_dich
        RETURN
          count(CASE WHEN co_nguon AND co_dich THEN 1 END) AS hop_le,
          count(CASE WHEN NOT co_nguon THEN 1 END)          AS thieu_nguon,
          count(CASE WHEN NOT co_dich  THEN 1 END)          AS thieu_dich
    """)
    r = rows[0]
    assert r["thieu_nguon"] == 0, (
        f"{r['thieu_nguon']} Action không có Component nguồn (HAS_ACTION). "
        "Kiểm tra load_action_edges()."
    )
    assert r["thieu_dich"] == 0, (
        f"{r['thieu_dich']} Action không có Component đích (APPLY_TO). "
        "Kiểm tra load_action_edges()."
    )


# ══════════════════════════════════════════════════════════════════════
# NHÓM 2 — RAG truy xuất được không?
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.critical
def test_tc04_vector_index_day_du(driver):
    """TC-04: Vector index tồn tại trên TextUnit và populationPercent = 100."""
    if not _vector_index_exists(driver):
        pytest.skip(
            f"Vector index '{_VECTOR_INDEX_NAME}' chưa tồn tại. "
            "Tạo thủ công theo hướng dẫn trong README.md trước khi chạy."
        )
    rows = _q(driver, "SHOW VECTOR INDEXES")
    idx = next((r for r in rows if r.get("name") == _VECTOR_INDEX_NAME), None)
    assert idx is not None

    state = idx.get("state", "")
    assert state == "ONLINE", f"Vector index ở trạng thái '{state}' — cần ONLINE."

    pop = idx.get("populationPercent", 0)
    assert pop >= 99.0, (
        f"Vector index mới index {pop:.1f}% (kỳ vọng 100%). "
        "Chờ Neo4j hoàn tất index hoặc kiểm tra có TextUnit nào bị bỏ sót."
    )


@pytest.mark.critical
def test_tc05_traversal_nguoc_len_norm(driver):
    """TC-05: TextUnit ngẫu nhiên (mẫu 20) phải đi ngược lên Norm và có số hiệu."""
    rows = _q(driver, """
        MATCH (tu:TextUnit)
        WHERE tu.type <> 'cache_action' AND tu.embedding IS NOT NULL
        WITH tu, rand() AS r ORDER BY r LIMIT 20
        MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu)
        OPTIONAL MATCH (n:Norm)-[:CONTAINS*]->(c)
        RETURN
          count(tu)                                             AS tong_mau,
          count(CASE WHEN n IS NOT NULL THEN 1 END)             AS len_duoc_norm,
          count(CASE WHEN n.norm_number IS NOT NULL THEN 1 END) AS co_so_hieu
    """)
    r = rows[0]
    if r["tong_mau"] == 0:
        pytest.skip("Không có TextUnit nào được embed — chạy --stage embed trước")
    assert r["len_duoc_norm"] == r["tong_mau"], (
        f"{r['tong_mau'] - r['len_duoc_norm']}/{r['tong_mau']} TextUnit không truy ngược "
        "lên được Norm. Liên quan TC-02 — kiểm tra cây không đứt gãy."
    )
    assert r["co_so_hieu"] == r["tong_mau"], (
        f"{r['tong_mau'] - r['co_so_hieu']}/{r['tong_mau']} Norm không có norm_number. "
        "Kiểm tra load_norms() — cột so_ky_hieu có bị bỏ sót không."
    )


@pytest.mark.critical
def test_tc06_vector_search_end_to_end(driver):
    """TC-06: Embed câu hỏi → tìm index → kết quả liên quan, score > 0.7."""
    if not _vector_index_exists(driver):
        pytest.skip(f"Vector index '{_VECTOR_INDEX_NAME}' chưa tồn tại")

    from eval.retriever import is_embed_available, retrieve
    if not is_embed_available():
        pytest.skip("GCP_PROJECT chưa set — không embed được câu hỏi")

    ket_qua = retrieve(driver, "điều kiện để được bầu cử đại biểu Quốc hội", top_k=5)
    if ket_qua is None:
        pytest.skip("Vertex AI không khả dụng — không embed được câu hỏi")

    assert len(ket_qua) >= 3, f"Chỉ tìm được {len(ket_qua)} kết quả (kỳ vọng ≥3)"
    assert any(
        "bầu cử" in (r.get("text") or "").lower()
        or "ứng cử" in (r.get("text") or "").lower()
        for r in ket_qua
    ), "Kết quả không liên quan đến câu hỏi về bầu cử"
    assert ket_qua[0]["score"] > 0.7, (
        f"Score cao nhất chỉ {ket_qua[0]['score']:.3f} (kỳ vọng >0.7)"
    )

    for r in ket_qua:
        print(f"  score={r['score']:.3f}  {(r.get('norm_title') or '')[:50]} — {r.get('citation', '')}")


# ══════════════════════════════════════════════════════════════════════
# NHÓM 3 — Nội dung đúng không?
# ══════════════════════════════════════════════════════════════════════


def test_tc07_textunit_khong_rong(driver):
    """TC-07: < 1% TextUnit có accumulated_text rỗng hoặc quá ngắn (<20 ký tự).

    TextUnit rỗng sinh ra vector vô nghĩa, làm nhiễu kết quả tìm kiếm.
    """
    rows = _q(driver, """
        MATCH (tu:TextUnit)
        WHERE tu.type <> 'cache_action'
        WITH count(tu) AS tong
        CALL {
          MATCH (tu:TextUnit)
          WHERE tu.type <> 'cache_action'
            AND (tu.accumulated_text IS NULL OR size(trim(tu.accumulated_text)) < 20)
          RETURN count(tu) AS so_rong, collect(tu.unit_id)[..5] AS vi_du
        }
        RETURN tong, so_rong, vi_du
    """)
    r = rows[0]
    if r["tong"] == 0:
        pytest.skip("Không có TextUnit")
    ty_le = 100.0 * r["so_rong"] / r["tong"]
    assert ty_le < 1.0, (
        f"{ty_le:.2f}% TextUnit rỗng/quá ngắn (ngưỡng: <1%). "
        f"Ví dụ unit_id: {r['vi_du']}. "
        "Kiểm tra text_accumulator."
    )


def test_tc08_citation_dung_chieu(driver):
    """TC-08: 0 Component có citation path ngược chiều ('Mục X > Chương Y').

    Lỗi từng xuất hiện thực tế — citation bị đảo ngược khiến action_extractor
    không khớp được Component B.
    """
    rows = _q(driver, r"""
        MATCH (c:Component)
        WHERE c.citation =~ '.*Mục\s+\d+.*>.*Chương.*'
        RETURN count(c) AS citation_nguoc_chieu, collect(c.citation)[..5] AS vi_du
    """)
    r = rows[0]
    assert r["citation_nguoc_chieu"] == 0, (
        f"{r['citation_nguoc_chieu']} Component có citation ngược chiều: {r['vi_du']}. "
        "Lỗi đã fix trong build_ancestor_chain() — check lại phiên bản code."
    )
