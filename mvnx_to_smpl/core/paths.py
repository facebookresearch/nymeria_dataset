# core/paths.py

from pathlib import Path

class Paths:
    PROJECT_ROOT = Path(__file__).parent.parent
    SMPL_FILE = PROJECT_ROOT / "assets" / "smpl" / "basicmodel_m.pkl"