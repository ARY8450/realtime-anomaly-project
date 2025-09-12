import sys
from pathlib import Path

# Ensure repository root is on sys.path before pytest collects tests
ROOT = Path(__file__).resolve().parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
