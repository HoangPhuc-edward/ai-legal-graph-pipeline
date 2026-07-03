"""load_norms / load_components / load_component_textunits / load_actions /
load_action_edges / load_relations + load_with_limit (guard ngưỡng AuraDB Free).

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

import logging

from schema.edges import NormRelation
from schema.nodes import Action, Component, Norm, TextUnit

from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def load_norms(client: Neo4jClient, norms: list[Norm]) -> None:
    logger.info("Load Norm: %d văn bản...", len(norms))
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
    logger.info("Load Norm xong: %d node :Norm", len(norms))


def load_components(client: Neo4jClient, components: list[Component]) -> None:
    logger.info("Load Component: %d component...", len(components))
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
    logger.info(
        "Load Component xong: %d node :Component, %d cạnh CONTAINS gốc, %d cạnh CONTAINS lồng",
        len(components), len(root_edges), len(nested_edges),
    )


def load_component_textunits(
    client: Neo4jClient,
    text_units: list[TextUnit],
    component_owner_map: dict[str, str],
) -> None:
    """text_units: chỉ TextUnit type="noi_dung" (sở hữu bởi Component).
    component_owner_map: comp_id -> unit_id."""
    logger.info("Load TextUnit (nội dung): %d unit...", len(text_units))
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
    logger.info(
        "Load TextUnit (nội dung) xong: %d node :TextUnit, %d cạnh HAS_TEXTUNIT (Component→TextUnit)",
        len(text_units), len(edges),
    )


def load_actions(client: Neo4jClient, actions: list[Action]) -> None:
    """Chỉ tạo node Action — KHÔNG kèm cạnh (cạnh HAS_ACTION/APPLY_TO tạo ở
    load_action_edges(), vì cần cả Component A và Component B đã tồn tại)."""
    logger.info("Load Action (node): %d action...", len(actions))
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
    logger.info("Load Action xong: %d node :Action", len(actions))


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
    logger.info("Load Action edges + TextUnit cache: %d action link...", len(action_links))
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
    logger.info(
        "Load Action edges xong: %d cặp HAS_ACTION + APPLY_TO + %d TextUnit cache :HAS_TEXTUNIT",
        len(rows), len(rows),
    )


class _LimitReached(Exception):
    """Internal sentinel — dừng load khi sẽ vượt ngưỡng AuraDB Free."""


def _trim_to_node_budget(
    items: list,
    item_count_fn,
    client: Neo4jClient,
    max_nodes: int,
    label: str,
) -> list:
    """Cắt `items` sao cho tổng node hiện có + số node mới không vượt max_nodes.

    item_count_fn(items) -> int: số node mới sẽ tạo từ items này (có thể khác
    len(items) vì 1 load step có thể sinh edge lẫn node).
    """
    current = client.count_nodes()
    budget = max_nodes - current
    new_count = item_count_fn(items)
    if new_count <= budget:
        return items
    if budget <= 0:
        logger.warning("[--limit-aura] %s: đã đạt ngưỡng %d node — bỏ qua toàn bộ.", label, max_nodes)
        return []
    cut = min(budget, len(items))
    logger.warning(
        "[--limit-aura] %s: chỉ load %d/%d item (còn budget %d node, ngưỡng %d).",
        label, cut, len(items), budget, max_nodes,
    )
    return items[:cut]


def _trim_to_edge_budget(
    items: list,
    edges_per_item: int,
    client: Neo4jClient,
    max_edges: int,
    label: str,
) -> list:
    current = client.count_edges()
    budget = max_edges - current
    if budget <= 0:
        logger.warning("[--limit-aura] %s: đã đạt ngưỡng %d edge — bỏ qua toàn bộ.", label, max_edges)
        return []
    cut = min(len(items), budget // max(edges_per_item, 1))
    if cut < len(items):
        logger.warning(
            "[--limit-aura] %s: chỉ load %d/%d item (còn budget %d edge, ngưỡng %d).",
            label, cut, len(items), budget, max_edges,
        )
    return items[:cut]


def load_with_limit(
    client: Neo4jClient,
    norms: list[Norm],
    components: list[Component],
    component_text_units: list[TextUnit],
    component_textunit_map: dict[str, str],
    actions: list[Action],
    action_links: list[dict],
    cache_text_units: dict[str, TextUnit],
    relations: list[NormRelation],
    max_nodes: int = 200_000,
    max_edges: int = 400_000,
) -> None:
    """Wrapper quanh các hàm load_* với guard AuraDB Free.

    Trước MỖI bước load, đếm node/edge hiện có trong Neo4j rồi cắt batch nếu
    sẽ vượt ngưỡng. Dừng sạch (log warning, không crash) thay vì âm thầm bỏ qua
    hay raise exception không rõ ràng.

    Ưu tiên thứ tự: Norm → Component → TextUnit → Action → Edge (giữ nguyên
    dependency hiện tại để tránh orphan edge).
    """
    norms_cut = _trim_to_node_budget(norms, len, client, max_nodes, "Norm")
    load_norms(client, norms_cut)

    components_cut = _trim_to_node_budget(components, len, client, max_nodes, "Component")
    load_components(client, components_cut)

    # TextUnit noi_dung: 1 node + 1 edge (HAS_TEXTUNIT) mỗi item — guard node trước
    tu_cut = _trim_to_node_budget(component_text_units, len, client, max_nodes, "TextUnit(noi_dung)")
    tu_cut = _trim_to_edge_budget(tu_cut, 1, client, max_edges, "HAS_TEXTUNIT(Component)")
    if tu_cut:
        load_component_textunits(client, tu_cut, component_textunit_map)

    actions_cut = _trim_to_node_budget(actions, len, client, max_nodes, "Action")
    load_actions(client, actions_cut)

    # action_links: 1 node TextUnit cache + 3 edge (HAS_ACTION, APPLY_TO, HAS_TEXTUNIT) mỗi link
    links_cut = _trim_to_node_budget(action_links, len, client, max_nodes, "TextUnit(cache_action)")
    links_cut = _trim_to_edge_budget(links_cut, 3, client, max_edges, "HAS_ACTION+APPLY_TO+HAS_TEXTUNIT")
    if links_cut:
        load_action_edges(client, links_cut, cache_text_units)

    # NormRelation: chỉ edge (Norm đã có sẵn từ bước Norm trên)
    relations_cut = _trim_to_edge_budget(relations, 1, client, max_edges, "NormRelation")
    if relations_cut:
        load_relations(client, relations_cut)

    final_nodes = client.count_nodes()
    final_edges = client.count_edges()
    logger.info("[--limit-aura] Load xong: %d node, %d edge (ngưỡng %d/%d)", final_nodes, final_edges, max_nodes, max_edges)


def load_relations(client: Neo4jClient, relations: list[NormRelation]) -> None:
    """Cạnh Tầng A: (Norm)-[:RELATION_TYPE]->(Norm) — nhãn cạnh động theo
    RelationType. Cypher không tham số hoá được relationship type, nhưng
    relation_type luôn xuất phát từ enum cố định (10 giá trị) nên an toàn để
    nội suy trực tiếp vào câu lệnh — nhóm theo loại, 1 query/loại."""
    logger.info("Load NormRelation: %d cạnh Norm→Norm...", len(relations))
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
        logger.info("  %s: %d cạnh", relation_type, len(rows))
    logger.info("Load NormRelation xong: %d cạnh tổng, %d loại", len(relations), len(by_type))
