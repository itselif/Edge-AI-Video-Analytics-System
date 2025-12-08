# tests/conftest.py
from pathlib import Path
import sys

# Proje kökü: Edge-AI-Video-Analytics-System
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)

if root_str not in sys.path:
    sys.path.insert(0, root_str)
