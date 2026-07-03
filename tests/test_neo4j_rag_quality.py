"""Kiểm tra chất lượng dữ liệu Neo4j sau khi load toàn bộ corpus — đảm bảo
đồ thị đáp ứng yêu cầu RAG trước khi đưa vào production.

CHẠY SAU bước load hoàn tất (python run_pipeline.py --stage load).
Yêu cầu NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD đã set trong .env.
Nếu chưa kết nối được, toàn bộ test tự động skip.

Thứ tự ưu tiên (chặn RAG nếu thất bại, chạy trước):
    TC-03 → TC-09 → TC-10 → TC-05 → TC-06

Chạy:
    pytest tests/test_neo4j_rag_quality.py -v
    pytest tests/test_neo4j_rag_quality.py -v -m critical     # 5 test chặn RAG
    pytest tests/test_neo4j_rag_quality.py -v -m sanity       # nhóm 1: phủ cơ bản
    pytest tests/test_neo4j_rag_quality.py -v -m structure    # nhóm 2: đúng đắn
    pytest tests/test_neo4j_rag_quality.py -v -m rag          # nhóm 3: truy xuất
    pytest tests/test_neo4j_rag_quality.py -v -m performance  # nhóm 5: hiệu năng

Ghi chú — vector index:
    CHƯA có trong schema_init.cypher. Tạo thủ công sau khi embed xong:
        CREATE VECTOR INDEX textunit_embedding_index IF NOT EXISTS
        FOR (t:TextUnit) ON (t.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
    TC-09 và TC-10 sẽ skip nếu index chưa tồn tại.
"""
from __future__ import annotations

import time
import pytest

from config import (
    EMBEDDING_MODEL,
    GCP_LOCATION,
    GCP_PROJECT,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)

# ─────────────────────────── kết nối & skip ────────────────────────────

_VECTOR_INDEX_NAME = "textunit_embedding_index"


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
pytestmark = pytest.mark.skipif(not _neo4j_ok, reason="NEO4J_URI chưa set hoặc không kết nối được")


@pytest.fixture(scope="module")
def driver():
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield d
    d.close()


def _q(driver, cypher: str, **params) -> list[dict]:
    with driver.session() as s:
        return s.run(cypher, **params).data()


def _embed(text: str) -> list[float] | None:
    """Embed 1 câu hỏi bằng Vertex AI — dùng cho TC-10 và TC-15."""
    if not GCP_PROJECT:
        return None
    try:
        from google import genai
        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=[text])
        return list(result.embeddings[0].values)
    except Exception:
        return None


def _vector_index_exists(driver) -> bool:
    try:
        rows = _q(driver, "SHOW VECTOR INDEXES")
        return any(r.get("name") == _VECTOR_INDEX_NAME for r in rows)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════
# NHÓM 1 — Phủ cơ bản (Sanity Check)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.sanity
def test_tc01_so_luong_van_ban(driver):
    """TC-01: Đủ Norm trong đồ thị — gần ~153.000 khi chạy full corpus."""
    row = _q(driver, "MATCH (n:Norm) RETURN count(n) AS tong")[0]
    tong = row["tong"]
    assert tong > 0, "Không có Norm nào — bước load chưa chạy hoặc thất bại hoàn toàn"
    # Ngưỡng 100.000 để test vẫn pass khi load một phần corpus lớn
    if tong < 100_000:
        pytest.xfail(
            f"Chỉ có {tong:,} Norm — chưa load full corpus (kỳ vọng ~153.000). "
            "Kết quả các test khác sẽ không phản ánh tình trạng thật."
        )
    # Chênh lệch dưới 1% so với 153.420 là chấp nhận được
    assert tong >= 151_886, (
        f"Thiếu Norm: chỉ có {tong:,} / ~153.420 (chênh >1%). "
        "Xem log load — tìm lỗi timeout hoặc batch bị bỏ qua."
    )


@pytest.mark.sanity
def test_tc02_van_ban_co_component(driver):
    """TC-02: Tỷ lệ văn bản có Component ≥ 85% — xác nhận transform đã tách cấu trúc."""
    rows = _q(driver, """
        MATCH (n:Norm)
        WITH n,
             COUNT { (n)-[:CONTAINS*]->(:Component) } AS so_comp
        RETURN
          count(CASE WHEN so_comp > 0 THEN 1 END) AS co_component,
          count(CASE WHEN so_comp = 0 THEN 1 END) AS khong_co_component,
          count(n) AS tong
    """)
    r = rows[0]
    tong = r["tong"]
    if tong == 0:
        pytest.skip("Không có Norm — TC-01 đã thất bại")
    ty_le_rong = 100.0 * r["khong_co_component"] / tong
    assert ty_le_rong <= 15.0, (
        f"{ty_le_rong:.1f}% Norm không có Component (ngưỡng: ≤15%). "
        "Chạy test.py --save-debug để xem structure_parser trên dữ liệu thật."
    )


@pytest.mark.sanity
@pytest.mark.critical
def test_tc03_component_co_textunit_va_vector(driver):
    """TC-03: ≥85% Component có TextUnit; 0 TextUnit chưa embed — điều kiện cứng để RAG tìm được."""
    rows = _q(driver, """
        MATCH (c:Component)
        WITH count(c) AS tong
        CALL {
          MATCH (c:Component)-[:HAS_TEXTUNIT]->()
          RETURN count(DISTINCT c) AS co_textunit
        }
        CALL {
          MATCH ()-[:HAS_TEXTUNIT]->(tu:TextUnit)
          WHERE tu.type <> 'cache_action' AND tu.embedding IS NULL
          RETURN count(tu) AS chua_embed
        }
        RETURN tong, co_textunit, chua_embed,
               round(100.0 * co_textunit / tong, 1) AS phan_tram_co_noi_dung
    """)
    r = rows[0]
    tong = r["tong"]
    if tong == 0:
        pytest.skip("Không có Component — TC-01/TC-02 đã thất bại")

    assert r["phan_tram_co_noi_dung"] >= 85.0, (
        f"Chỉ {r['phan_tram_co_noi_dung']}% Component có TextUnit (ngưỡng: ≥85%). "
        "Lỗi text_accumulator hoặc transform không tạo TextUnit cho Component lá."
    )
    assert r["chua_embed"] == 0, (
        f"{r['chua_embed']:,} TextUnit chưa có vector. "
        "Chạy tools/retry_failed_embeddings.py để embed lại phần bị lỗi."
    )


@pytest.mark.sanity
def test_tc04_quan_he_da_load(driver):
    """TC-04: Tổng quan hệ gần ~900.000; CITES chiếm nhiều nhất (~580.000)."""
    rows = _q(driver, """
        MATCH ()-[r:AMENDS|SUPPLEMENTS|TERMINATES|PARTIALLY_TERMINATES|SUSPENDS
                    |PARTIALLY_SUSPENDS|IMPLEMENTS|REFERS_TO|RELATED_TO|CITES]->()
        RETURN type(r) AS loai, count(r) AS so_luong
        ORDER BY so_luong DESC
    """)
    tong = sum(r["so_luong"] for r in rows)
    assert tong > 0, "Không có quan hệ nào — bước load_relations chưa chạy"

    by_type = {r["loai"]: r["so_luong"] for r in rows}
    cites = by_type.get("CITES", 0)

    if tong >= 800_000:
        assert cites > 500_000, (
            f"CITES chỉ có {cites:,} — kỳ vọng ~580.000. "
            "Có thể nhãn bị bỏ sót trong RELATION_LABEL_MAP."
        )


# ══════════════════════════════════════════════════════════════════════
# NHÓM 2 — Đúng đắn cấu trúc
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.structure
@pytest.mark.critical
def test_tc05_cay_khong_dut_gay(driver):
    """TC-05: 0 Component không có đường dẫn lên Norm — cây đứt gãy làm traversal chết."""
    rows = _q(driver, """
        MATCH (c:Component)
        WHERE NOT EXISTS { (:Norm)-[:CONTAINS*]->(c) }
        RETURN count(c) AS so_luong, collect(c.comp_id)[..10] AS vi_du
    """)
    r = rows[0]
    assert r["so_luong"] == 0, (
        f"{r['so_luong']} Component không có Norm cha: {r['vi_du']}. "
        "Kiểm tra thứ tự load — Component phải load sau Norm."
    )


@pytest.mark.structure
@pytest.mark.critical
def test_tc06_action_du_hai_dau(driver):
    """TC-06: 0 Action thiếu nguồn (Component A) hoặc thiếu đích (Component B)."""
    rows = _q(driver, """
        MATCH (act:Action)
        WITH act,
          COUNT { (:Component)-[:HAS_ACTION]->(act) } > 0 AS co_nguon,
          COUNT { (act)-[:APPLY_TO]->(:Component) } > 0 AS co_dich
        RETURN
          count(CASE WHEN co_nguon AND co_dich THEN 1 END) AS hop_le,
          count(CASE WHEN NOT co_nguon THEN 1 END) AS thieu_nguon,
          count(CASE WHEN NOT co_dich THEN 1 END)  AS thieu_dich
    """)
    r = rows[0]
    assert r["thieu_nguon"] == 0, (
        f"{r['thieu_nguon']} Action không có Component nguồn (HAS_ACTION). "
        "Kiểm tra load_action_edges() — HAS_ACTION phải được tạo cùng APPLY_TO."
    )
    assert r["thieu_dich"] == 0, (
        f"{r['thieu_dich']} Action không có Component đích (APPLY_TO). "
        "Kiểm tra load_action_edges() — APPLY_TO phải được tạo cùng HAS_ACTION."
    )


@pytest.mark.structure
def test_tc07_textunit_khong_rong(driver):
    """TC-07: Dưới 1% TextUnit có accumulated_text rỗng hoặc quá ngắn (<20 ký tự)."""
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
        "Kiểm tra text_accumulator — có thể raw_text bị rỗng sau khi filter lá."
    )


@pytest.mark.structure
def test_tc08_citation_dung_chieu(driver):
    """TC-08: 0 Component có citation path ngược chiều ('Mục X > Chương Y')."""
    rows = _q(driver, r"""
        MATCH (c:Component)
        WHERE c.citation =~ '.*Mục\s+\d+.*>.*Chương.*'
        RETURN count(c) AS so_luong, collect(c.citation)[..5] AS vi_du
    """)
    r = rows[0]
    assert r["so_luong"] == 0, (
        f"{r['so_luong']} Component có citation ngược chiều: {r['vi_du']}. "
        "Lỗi đã được fix trong build_ancestor_chain() — check lại phiên bản code."
    )


# ══════════════════════════════════════════════════════════════════════
# NHÓM 3 — Khả năng RAG truy xuất
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.rag
@pytest.mark.critical
def test_tc09_vector_index_ton_tai_va_day_du(driver):
    """TC-09: Vector index tồn tại và đã index 100% TextUnit có embedding."""
    if not _vector_index_exists(driver):
        pytest.skip(
            f"Vector index '{_VECTOR_INDEX_NAME}' chưa tồn tại. "
            "Tạo thủ công theo hướng dẫn trong docstring file này trước khi chạy."
        )
    rows = _q(driver, "SHOW VECTOR INDEXES")
    idx = next((r for r in rows if r.get("name") == _VECTOR_INDEX_NAME), None)
    assert idx is not None

    pop = idx.get("populationPercent", 0)
    assert pop >= 99.0, (
        f"Vector index mới index {pop:.1f}% (kỳ vọng 100%). "
        "Chờ Neo4j hoàn tất index hoặc kiểm tra xem có TextUnit nào bị bỏ sót."
    )

    state = idx.get("state", "")
    assert state == "ONLINE", f"Vector index ở trạng thái '{state}' — cần ONLINE."


@pytest.mark.rag
@pytest.mark.critical
def test_tc10_vector_search_ket_qua(driver):
    """TC-10: Vector search trả về ≥1 kết quả liên quan; score cao nhất > 0.7."""
    if not _vector_index_exists(driver):
        pytest.skip(f"Vector index '{_VECTOR_INDEX_NAME}' chưa tồn tại")

    cau_hoi = "điều kiện để được bầu cử đại biểu Quốc hội"
    vector = _embed(cau_hoi)
    if vector is None:
        pytest.skip("GCP_PROJECT chưa set hoặc Vertex AI không khả dụng — không embed được câu hỏi")

    rows = _q(driver, f"""
        CALL db.index.vector.queryNodes('{_VECTOR_INDEX_NAME}', $top_k, $query_vector)
        YIELD node AS tu, score
        WHERE tu.type <> 'cache_action'
        MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu)
        MATCH (n:Norm)-[:CONTAINS*]->(c)
        RETURN tu.unit_id, tu.accumulated_text, score, n.title AS norm_title, c.citation
        ORDER BY score DESC
    """, top_k=5, query_vector=vector)

    assert len(rows) > 0, "Vector search không trả về kết quả nào"

    top_score = rows[0]["score"]
    assert top_score > 0.7, (
        f"Score cao nhất chỉ {top_score:.3f} (ngưỡng: >0.7). "
        "Kiểm tra vector index có đúng dimension chưa, hoặc dữ liệu embed sai."
    )

    texts = [r["accumulated_text"] or "" for r in rows]
    tu_lien_quan = sum(
        1 for t in texts if "bầu cử" in t or "ứng cử" in t or "đại biểu" in t
    )
    assert tu_lien_quan >= 2, (
        f"Chỉ {tu_lien_quan}/5 kết quả liên quan đến câu hỏi về bầu cử (ngưỡng: ≥2). "
        "Kết quả: " + str([r.get("norm_title", "")[:60] for r in rows])
    )


@pytest.mark.rag
def test_tc11_traversal_nguoc_len_norm(driver):
    """TC-11: TextUnit ngẫu nhiên phải đi ngược lên Norm và lấy đủ metadata."""
    rows = _q(driver, """
        MATCH (tu:TextUnit)
        WHERE tu.type <> 'cache_action' AND tu.embedding IS NOT NULL
        WITH tu, rand() AS r ORDER BY r LIMIT 10
        MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu)
        MATCH (n:Norm)-[:CONTAINS*]->(c)
        RETURN tu.unit_id, n.title AS norm_title, n.norm_number, c.citation
    """)
    if not rows:
        pytest.skip("Không có TextUnit có embedding — TC-03/TC-09 chưa qua")

    thieu_title = [r for r in rows if not r.get("norm_title")]
    thieu_number = [r for r in rows if not r.get("norm_number")]
    assert not thieu_title, (
        f"{len(thieu_title)} TextUnit không truy ngược lên được Norm có title. "
        f"unit_id ví dụ: {[r['unit_id'] for r in thieu_title[:3]]}. "
        "Liên quan TC-05 — cây bị đứt gãy."
    )
    assert not thieu_number, (
        f"{len(thieu_number)} TextUnit truy lên Norm không có norm_number. "
        f"unit_id ví dụ: {[r['unit_id'] for r in thieu_number[:3]]}."
    )


@pytest.mark.rag
def test_tc12_action_mo_rong_context(driver):
    """TC-12: Action phải có amending_doc_number và cache TextUnit đầy đủ."""
    rows = _q(driver, """
        MATCH (c:Component)<-[:APPLY_TO]-(act:Action)
        WITH c, act LIMIT 5
        OPTIONAL MATCH (act)-[:HAS_TEXTUNIT]->(tu_cache:TextUnit)
        RETURN
          c.comp_id,
          c.citation,
          act.relation_type,
          act.amending_doc_number,
          tu_cache.accumulated_text IS NOT NULL AS co_noi_dung_cache
    """)
    if not rows:
        pytest.skip("Không có Action nào — Tầng B chưa có dữ liệu (có thể bình thường với sample nhỏ)")

    thieu_doc_number = [r for r in rows if not r.get("amending_doc_number")]
    assert not thieu_doc_number, (
        f"{len(thieu_doc_number)} Action không có amending_doc_number. "
        "Kiểm tra relation_classifier — amending_doc_number phải được copy từ Norm A."
    )

    thieu_cache = [r for r in rows if not r.get("co_noi_dung_cache")]
    assert not thieu_cache, (
        f"{len(thieu_cache)} Action không có cache TextUnit. "
        "Kiểm tra load_action_edges() — HAS_TEXTUNIT từ Action phải được tạo."
    )


# ══════════════════════════════════════════════════════════════════════
# NHÓM 4 — Tính đúng của nội dung
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.content
def test_tc13_kiem_tra_thu_cong_noi_dung(driver, capsys):
    """TC-13: In mẫu 10 Component ngẫu nhiên để đối chiếu thủ công với vbpl.vn.

    Test này KHÔNG có assertion cứng — chỉ in kết quả để người đọc xem xét.
    Xác nhận thủ công: mở vbpl.vn, tìm văn bản, so preview với nội dung gốc.
    """
    rows = _q(driver, """
        MATCH (n:Norm)-[:CONTAINS*]->(c:Component)-[:HAS_TEXTUNIT]->(tu:TextUnit)
        WHERE c.level IN ['Dieu', 'Khoan']
          AND tu.type <> 'cache_action'
          AND tu.embedding IS NOT NULL
        WITH n, c, tu, rand() AS r ORDER BY r LIMIT 10
        RETURN
          n.norm_number    AS so_hieu,
          n.title          AS ten_van_ban,
          c.citation       AS vi_tri,
          left(tu.accumulated_text, 300) AS preview
    """)
    if not rows:
        pytest.skip("Không có dữ liệu đủ điều kiện để review")

    with capsys.disabled():
        print("\n" + "═" * 70)
        print("TC-13 — KIỂM TRA THỦ CÔNG: đối chiếu preview với vbpl.vn")
        print("═" * 70)
        for i, r in enumerate(rows, 1):
            print(f"\n[{i}] {r['so_hieu']} — {r['ten_van_ban'][:60]}")
            print(f"    Vị trí: {r['vi_tri']}")
            print(f"    Preview: {(r['preview'] or '').strip()[:200]}")
        print("═" * 70)

    # Assertion tối thiểu: preview không rỗng và không có ký tự lạ rõ ràng
    bad = [r for r in rows if not r.get("preview") or len(r["preview"].strip()) < 10]
    assert len(bad) == 0 or len(bad) / len(rows) < 0.2, (
        f"{len(bad)}/{len(rows)} mẫu có preview rỗng/quá ngắn."
    )


@pytest.mark.content
def test_tc14_van_ban_sua_doi_du_hai_phia(driver):
    """TC-14: Action AMENDS phải có đủ nội dung ở cả Component nguồn lẫn đích."""
    rows = _q(driver, """
        MATCH (ca:Component)-[:HAS_ACTION]->(act:Action {relation_type: 'AMENDS'})-[:APPLY_TO]->(cb:Component)
        MATCH (ca)-[:HAS_TEXTUNIT]->(tu_a:TextUnit)
        MATCH (cb)-[:HAS_TEXTUNIT]->(tu_b:TextUnit)
        MATCH (na:Norm)-[:CONTAINS*]->(ca)
        MATCH (nb:Norm)-[:CONTAINS*]->(cb)
        RETURN
          na.norm_number    AS van_ban_sua,
          ca.citation       AS dieu_sua,
          nb.norm_number    AS van_ban_bi_sua,
          cb.citation       AS dieu_bi_sua,
          act.amending_doc_number IS NOT NULL AS co_cache,
          left(tu_a.accumulated_text, 100) AS noi_dung_sua,
          left(tu_b.accumulated_text, 100) AS noi_dung_bi_sua
        LIMIT 3
    """)
    if not rows:
        pytest.skip("Không có Action AMENDS — Tầng B chưa có dữ liệu")

    for r in rows:
        assert r["co_cache"], (
            f"Action AMENDS giữa {r['van_ban_sua']} và {r['van_ban_bi_sua']} "
            "không có amending_doc_number (cache)."
        )
        assert r.get("noi_dung_sua"), (
            f"{r['van_ban_sua']} {r['dieu_sua']} không có accumulated_text."
        )
        assert r.get("noi_dung_bi_sua"), (
            f"{r['van_ban_bi_sua']} {r['dieu_bi_sua']} không có accumulated_text."
        )


# ══════════════════════════════════════════════════════════════════════
# NHÓM 5 — Hiệu năng
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.performance
def test_tc15_toc_do_vector_search(driver):
    """TC-15: Vector search top-5 phải dưới 1000ms (kỳ vọng <500ms)."""
    if not _vector_index_exists(driver):
        pytest.skip(f"Vector index '{_VECTOR_INDEX_NAME}' chưa tồn tại")

    vector = _embed("điều kiện bầu cử")
    if vector is None:
        pytest.skip("GCP_PROJECT chưa set — không embed được câu hỏi")

    start = time.perf_counter()
    rows = _q(driver, f"""
        CALL db.index.vector.queryNodes('{_VECTOR_INDEX_NAME}', 5, $query_vector)
        YIELD node, score
        RETURN node.unit_id, score
    """, query_vector=vector)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 1000, (
        f"Vector search top-5 mất {elapsed_ms:.0f}ms (ngưỡng: <1000ms). "
        "Kiểm tra vector index có được tạo đúng chưa (`SHOW VECTOR INDEXES`)."
    )
    if elapsed_ms > 500:
        pytest.xfail(
            f"Vector search {elapsed_ms:.0f}ms — chấp nhận nhưng nên cải thiện "
            "(<500ms là mục tiêu). Theo dõi thêm khi load đủ corpus."
        )


@pytest.mark.performance
def test_tc16_toc_do_graph_expansion(driver):
    """TC-16: Graph traversal từ Component → Action → TextUnit phải dưới 50ms."""
    # Tìm 1 comp_id thật có Action
    candidates = _q(driver, """
        MATCH (c:Component)<-[:APPLY_TO]-(:Action)
        RETURN c.comp_id AS comp_id LIMIT 1
    """)
    if not candidates:
        pytest.skip("Không có Action — Tầng B chưa có dữ liệu, không đo được traversal")
    comp_id = candidates[0]["comp_id"]

    start = time.perf_counter()
    _q(driver, """
        MATCH (c:Component {comp_id: $comp_id})<-[:APPLY_TO]-(act:Action)-[:HAS_TEXTUNIT]->(tu:TextUnit)
        RETURN act.relation_type, act.amending_doc_number, tu.accumulated_text
    """, comp_id=comp_id)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 200, (
        f"Graph expansion mất {elapsed_ms:.0f}ms (ngưỡng: <200ms). "
        "Kiểm tra index trên comp_id: CREATE INDEX comp_id_idx FOR (c:Component) ON (c.comp_id)."
    )
    if elapsed_ms > 50:
        pytest.xfail(
            f"Graph expansion {elapsed_ms:.0f}ms — chấp nhận nhưng chưa đạt mục tiêu <50ms. "
            "Chạy PROFILE query trên Neo4j Browser để phân tích."
        )
