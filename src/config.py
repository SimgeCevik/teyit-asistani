"""Central configuration for the fact-check assistant."""

from pathlib import Path

# Paths are built relative to this file, so they work from notebooks, scripts and the app
# .resolve().parent.parent → the folder containing config.py (src), and the folder above that—in other words, the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FCTR_CLEAN_PATH = DATA_PROCESSED / "fctr_clean.parquet"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Models
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Vector store
COLLECTION_NAME = "fctr_chunks"

# Chunking
CHUNK_MAX_TOKENS = 300
CHUNK_OVERLAP_SENTENCES = 1

# Evaluation
EVAL_DIR = PROJECT_ROOT / "data" / "eval"  # raw evaluation outputs (not committed)
RESULTS_DIR = PROJECT_ROOT / "results" # summary metrics (committed)