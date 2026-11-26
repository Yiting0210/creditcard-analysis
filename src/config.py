"""Paths and connection settings, all overridable through environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", ROOT / "reports"))

# New applications to score each run (the Kaggle test set has no labels, so it stands in for new applicants).
INPUT_FILE = os.getenv("INPUT_FILE", "application_test.csv")

# SQLAlchemy URL, e.g. postgresql+psycopg2://user:pass@host:5432/homecredit
DATABASE_URL = os.getenv("DATABASE_URL", "")

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
