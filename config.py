"""Đọc .env, hằng số dùng chung, bảng rank cấp văn bản (Component level)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Neo4j AuraDB ---
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# --- Google Cloud / Vertex AI ---
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
LLM_MODEL_HEAVY = os.getenv("LLM_MODEL_HEAVY", "gemini-3.5-flash")
LLM_MODEL_LIGHT = os.getenv("LLM_MODEL_LIGHT", "gemini-2.5-flash")

# --- Hugging Face dataset ---
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "th1nhng0/vietnamese-legal-documents")

# --- Đường dẫn dữ liệu trung gian ---
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
RAW_DIR = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
TRANSFORMED_DIR = DATA_DIR / "transformed"
EMBEDDED_DIR = DATA_DIR / "embedded"

for _dir in (RAW_DIR, FILTERED_DIR, TRANSFORMED_DIR, EMBEDDED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Bảng rank cấp văn bản (Component) — số nhỏ hơn = cấp cao hơn (nông hơn trong cây) ---
LEVEL_RANK = {
    "Phan": 0,
    "Chuong": 1,
    "Muc": 2,
    "TieuMuc": 3,
    "Dieu": 4,
    "Khoan": 5,
    "Diem": 6,
}

# --- Batch size dùng cho ghi Neo4j (UNWIND + MERGE) ---
NEO4J_BATCH_SIZE = 500
