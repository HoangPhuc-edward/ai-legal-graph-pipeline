"""load_norms / load_components / load_component_textunits / load_actions /
load_action_edges / load_relations.

Thứ tự load BẮT BUỘC (phụ thuộc node đã tồn tại từ bước trước):
  1. load_norms()
  2. load_components()          (+ CONTAINS: Norm->Component, Component->Component)
  3. load_component_textunits() (+ HAS_TEXTUNIT: Component->TextUnit, type="noi_dung")
  4. load_actions()              tạo node Action (gồm amending_doc_number — đã
                                  có sẵn lúc transform), KHÔNG kèm cạnh
  5. load_action_edges()         + HAS_ACTION (Component A->Action), APPLY_TO
                                  (Action->Component B), + cache TextUnit kèm
                                  HAS_TEXTUNIT (Action->TextUnit)             — Tầng B
  6. load_relations()            cạnh NormRelation (Norm->Norm trực tiếp)     — Tầng A

Bước 4 và 5 tách riêng vì Action cần CẢ HAI Component (A và B) đã tồn tại
trước khi nối cạnh — tách node-creation khỏi edge-creation tránh lỗi thứ tự
nếu Component A và Component B thuộc 2 văn bản được load ở 2 thời điểm khác
nhau. TextUnit cache của Action gộp chung vào bước 5 vì luôn đi kèm 1-1 với
việc tạo cạnh HAS_ACTION/APPLY_TO — cùng 1 transaction logic.
"""
from __future__ import annotations

from schema.edges import NormRelation
from schema.nodes import Action, Component, Norm, TextUnit

from .neo4j_client import Neo4jClient


def load_norms(client: Neo4jClient, norms: list[Norm]) -> None:
    cypher = """
    UNWIND $rows AS row
    MERGE (n:Norm {norm_id: row.norm_id})
    SET n.title = row.title,
        n.norm_number = row.norm_number,
        n.norm_type = row.norm_type,
        n.published_date = row.published_date,
        n.valid_from = row.valid_from,
        n.valid_to = row.valid_to,
        n.publisher = row.publisher,
        n.signer = row.signer,
        n.validity_status = row.validity_status,
        n.sector = row.sector,
        n.field = row.field,
        n.updated_at = row.updated_at
    """
    rows = [n.model_dump(mode="json") for n in norms]
    client.batch_write(cypher, rows)


def load_components(client: Neo4jClient, components: list[Component]) -> None:
    node_cypher = """
    UNWIND $rows AS row
    MERGE (c:Component {comp_id: row.comp_id})
    SET c.norm_id = row.norm_id,
        c.level = row.level,
        c.citation = row.citation,
        c.order_index = row.order_index,
        c.title_text = row.title_text,
        c.updated_at = row.updated_at
    """
    rows = [c.model_dump(mode="json") for c in components]
    client.batch_write(node_cypher, rows)

    # CONTAINS — Norm -> Component (gốc) hoặc Component -> Component (lồng nhau)
    root_edges = [
        {"parent_id": c.norm_id, "child_id": c.comp_id} for c in components if c.parent_comp_id is None
    ]
    nested_edges = [
        {"parent_id": c.parent_comp_id, "child_id": c.comp_id}
        for c in components
        if c.parent_comp_id is not None
    ]

    root_cypher = """
    UNWIND $rows AS row
    MATCH (p:Norm {norm_id: row.parent_id})
    MATCH (c:Component {comp_id: row.child_id})
    MERGE (p)-[:CONTAINS]->(c)
    """
    nested_cypher = """
    UNWIND $rows AS row
    MATCH (p:Component {comp_id: row.parent_id})
    MATCH (c:Component {comp_id: row.child_id})
    MERGE (p)-[:CONTAINS]->(c)
    """
    client.batch_write(root_cypher, root_edges)
    client.batch_write(nested_cypher, nested_edges)


def load_component_textunits(
    client: Neo4jClient,
    text_units: list[TextUnit],
    component_owner_map: dict[str, str],
) -> None:
    """text_units: chỉ TextUnit type="noi_dung" (sở hữu bởi Component).
    component_owner_map: comp_id -> unit_id."""
    node_cypher = """
    UNWIND $rows AS row
    MERGE (t:TextUnit {unit_id: row.unit_id})
    SET t.accumulated_text = row.accumulated_text,
        t.type = row.type,
        t.language = row.language,
        t.embedding = row.embedding,
        t.embedded_at = row.embedded_at,
        t.error_log = row.error_log,
        t.updated_at = row.updated_at
    """
    rows = [tu.model_dump(mode="json") for tu in text_units]
    client.batch_write(node_cypher, rows)

    unit_to_comp = {unit_id: comp_id for comp_id, unit_id in component_owner_map.items()}
    edges = [
        {"owner_id": unit_to_comp[tu.unit_id], "unit_id": tu.unit_id}
        for tu in text_units
        if tu.unit_id in unit_to_comp
    ]
    edge_cypher = """
    UNWIND $rows AS row
    MATCH (c:Component {comp_id: row.owner_id})
    MATCH (t:TextUnit {unit_id: row.unit_id})
    MERGE (c)-[:HAS_TEXTUNIT]->(t)
    """
    client.batch_write(edge_cypher, edges)


def load_actions(client: Neo4jClient, actions: list[Action]) -> None:
    """Chỉ tạo node Action — KHÔNG kèm cạnh (cạnh HAS_ACTION/APPLY_TO tạo ở
    load_action_edges(), vì cần cả Component A và Component B đã tồn tại)."""
    node_cypher = """
    UNWIND $rows AS row
    MERGE (a:Action {action_id: row.action_id})
    SET a.relation_type = row.relation_type,
        a.amending_doc_number = row.amending_doc_number,
        a.effective_date = row.effective_date,
        a.description = row.description,
        a.updated_at = row.updated_at
    """
    rows = [a.model_dump(mode="json") for a in actions]
    client.batch_write(node_cypher, rows)


def load_action_edges(
    client: Neo4jClient,
    action_links: list[dict],
    cache_text_units: dict[str, TextUnit],
) -> None:
    """Tầng B — kèm 1 lượt: HAS_ACTION (Component A -> Action), APPLY_TO
    (Action -> Component B), và TextUnit cache riêng của Action (+ HAS_TEXTUNIT).

    action_links: list các dict {action_id, comp_a_id, comp_b_id, cache_unit_id}.
    cache_text_units: cache_unit_id -> TextUnit (type="cache_action", embedding=None).
    """
    rows = [
        {
            "action_id": link["action_id"],
            "comp_a_id": link["comp_a_id"],
            "comp_b_id": link["comp_b_id"],
            "cache_unit_id": link["cache_unit_id"],
            "cache_text": cache_text_units[link["cache_unit_id"]].accumulated_text,
        }
        for link in action_links
        if link["cache_unit_id"] in cache_text_units
    ]
    cypher = """
    UNWIND $rows AS row
    MATCH (a:Component {comp_id: row.comp_a_id})
    MATCH (b:Component {comp_id: row.comp_b_id})
    MATCH (act:Action {action_id: row.action_id})
    MERGE (a)-[:HAS_ACTION]->(act)
    MERGE (act)-[:APPLY_TO]->(b)
    MERGE (act)-[:HAS_TEXTUNIT]->(tu:TextUnit {unit_id: row.cache_unit_id})
    SET tu.accumulated_text = row.cache_text, tu.type = 'cache_action', tu.embedding = null
    """
    client.batch_write(cypher, rows)


def load_relations(client: Neo4jClient, relations: list[NormRelation]) -> None:
    """Cạnh Tầng A: (Norm)-[:RELATION_TYPE]->(Norm) — nhãn cạnh động theo
    RelationType. Cypher không tham số hoá được relationship type, nhưng
    relation_type luôn xuất phát từ enum cố định (10 giá trị) nên an toàn để
    nội suy trực tiếp vào câu lệnh — nhóm theo loại, 1 query/loại."""
    by_type: dict[str, list[dict]] = {}
    for r in relations:
        by_type.setdefault(r.relation_type.value, []).append(
            {"from_norm_id": r.from_norm_id, "to_norm_id": r.to_norm_id}
        )

    for relation_type, rows in by_type.items():
        cypher = f"""
        UNWIND $rows AS row
        MATCH (a:Norm {{norm_id: row.from_norm_id}})
        MATCH (b:Norm {{norm_id: row.to_norm_id}})
        MERGE (a)-[:{relation_type}]->(b)
        """
        client.batch_write(cypher, rows)
