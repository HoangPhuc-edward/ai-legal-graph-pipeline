"""Health-check 5 điểm — chạy TRƯỚC mỗi run_pipeline.py stage.

Đặc biệt quan trọng trên Colab nơi token Vertex AI có thể hết hạn giữa chừng
(đúng lỗi 503 đã gặp khi chạy embed) — chạy health_check trước để phát hiện
sớm thay vì phát hiện giữa chừng sau nhiều giờ transform.

Chạy:
    python tools/health_check.py
    python tools/health_check.py --skip-llm   # bỏ qua check LLM/Embedding (nhanh hơn)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_OK = "[OK]"
_FAIL = "[FAIL]"
_SKIP = "[SKIP]"


def _ok(label: str, detail: str) -> None:
    print(f"{_OK}  {label:<20} — {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"{_FAIL} {label:<20} — {detail}")


def _skip(label: str) -> None:
    print(f"{_SKIP} {label:<20}")


def check_libraries() -> bool:
    packages = {
        "pyarrow": "pyarrow",
        "fast_html2md": "fast_html2md",
        "neo4j": "neo4j",
        "vertexai": "vertexai",
        "pydantic": "pydantic",
    }
    missing = []
    for name, pkg in packages.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(name)

    if missing:
        _fail("Libraries", f"thiếu package: {', '.join(missing)}")
        return False
    _ok("Libraries", "all imports OK")
    return True


def check_data_files() -> bool:
    import pyarrow.parquet as pq
    from config import RAW_DIR

    configs = ["metadata", "content", "relationships"]
    missing = []
    rows = {}
    for cfg in configs:
        path = RAW_DIR / f"{cfg}.parquet"
        if not path.exists():
            missing.append(cfg)
        else:
            pf = pq.ParquetFile(path)
            rows[cfg] = pf.metadata.num_rows

    if missing:
        _fail("Data files", f"chưa có: {', '.join(missing)} — chạy --stage extract trước")
        return False

    detail = ", ".join(f"{cfg}: {rows[cfg]:,}" for cfg in configs)
    _ok("Data files", detail)
    return True


def check_neo4j() -> bool:
    try:
        from load.neo4j_client import Neo4jClient
        with Neo4jClient() as client:
            with client._driver.session() as session:
                ver = session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
                nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        _ok("Neo4j", f"v{ver}, {nodes:,} nodes, {edges:,} edges")
        return True
    except Exception as exc:
        _fail("Neo4j", str(exc))
        return False


def check_gemini_llm() -> bool:
    try:
        from google import genai
        from config import GCP_LOCATION, GCP_PROJECT, LLM_MODEL_LIGHT

        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
        resp = client.models.generate_content(model=LLM_MODEL_LIGHT, contents="Reply with exactly: OK")
        text = (resp.text or "").strip()[:20]
        _ok("Gemini LLM", f"{LLM_MODEL_LIGHT} → {text!r}")
        return True
    except Exception as exc:
        _fail("Gemini LLM", str(exc))
        return False


def check_gemini_embedding() -> bool:
    try:
        from google import genai
        from config import EMBEDDING_MODEL, GCP_LOCATION, GCP_PROJECT

        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=["kiểm tra kết nối"])
        dim = len(result.embeddings[0].values)
        _ok("Embedding", f"{EMBEDDING_MODEL}, dim={dim}")
        return True
    except Exception as exc:
        _fail("Embedding", str(exc))
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Health-check 5 điểm trước khi chạy pipeline")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua check LLM + Embedding (tiết kiệm thời gian)")
    args = parser.parse_args()

    print("\n── Health Check ───────────────────────────────")
    passed = 0
    total = 0

    checks = [
        ("Libraries", check_libraries),
        ("Data files", check_data_files),
        ("Neo4j", check_neo4j),
    ]
    if not args.skip_llm:
        checks += [
            ("Gemini LLM", check_gemini_llm),
            ("Embedding", check_gemini_embedding),
        ]

    for label, fn in checks:
        total += 1
        ok = False
        try:
            ok = fn()
        except Exception as exc:
            _fail(label, f"exception không xử lý được: {exc}")
        if ok:
            passed += 1
        else:
            print(f"\n✗ Dừng lại — {label} FAIL. Sửa xong rồi chạy lại.\n")
            sys.exit(1)

    if args.skip_llm:
        _skip("Gemini LLM")
        _skip("Embedding")

    print("────────────────────────────────────────────────")
    print(f"✓ {passed}/{total} check passed — sẵn sàng chạy pipeline.\n")


if __name__ == "__main__":
    main()
