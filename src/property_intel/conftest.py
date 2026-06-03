# conftest.py (root level) — adds src/ to path so pytest can find property_intel

import sys
from pathlib import Path

# this runs before any test, adding src/ so imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))