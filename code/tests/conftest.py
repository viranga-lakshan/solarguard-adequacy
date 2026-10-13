"""Shared pytest fixtures. Adds the project's code/ directory to sys.path so
tests can `import load_mafate`, `import paths`, etc. the same way the
pipeline scripts themselves do, regardless of the directory pytest is
invoked from.
"""

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
