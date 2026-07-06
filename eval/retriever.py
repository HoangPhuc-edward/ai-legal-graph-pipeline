"""Vector search retriever — embed câu hỏi rồi query Neo4j vector index.

Dùng cho TC-06 (test_neo4j_rag_quality.py) và bất kỳ script eval nào cần
end-to-end retrieval. Không phụ thuộc vào pipeline code — chỉ dùng config + Neo4j driver.
"""
from __future__ import annotations

from typing import Optional

from config import EMBEDDING_MODEL, GCP_LOCATION, GCP_PROJECT

_VECTOR_INDEX_NAME = "textunit_embedding_index"


def is_embed_available() -> bool:
    """Trả True nếu GCP_PROJECT đã set — điều kiện cần để embed câu hỏi."""
    return bool(GCP_PROJECT)


def _embed(text: str) -> Optional[list[float]]:
    if not GCP_PROJECT:
        return None
    try:
        from google import genai
        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=[text])
        return list(result.embeddings[0].values)
    except Exception:
        return None


def retrieve(driver, query: str, top_k: int = 5) -> Optional[list[dict]]:
    """Embed câu hỏi và tìm TextUnit gần nhất qua vector index.

    Trả về list[dict] với keys: text, score, norm_title, citation.
    Trả None nếu không embed được (GCP chưa set hoặc Vertex AI lỗi).
    """
    vector = _embed(query)
    if vector is None:
        return None
    with driver.session() as session:
        rows = session.run(
            f"""
            CALL db.index.vector.queryNodes('{_VECTOR_INDEX_NAME}', $top_k, $query_vector)
            YIELD node AS tu, score
            WHERE tu.type <> 'cache_action'
            MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu)
            MATCH (n:Norm)-[:CONTAINS*]->(c)
            RETURN tu.accumulated_text AS text, score,
                   n.title AS norm_title, c.citation AS citation
            ORDER BY score DESC
            """,
            top_k=top_k,
            query_vector=vector,
        ).data()
    return rows
