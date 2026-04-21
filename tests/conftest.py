# tests/conftest.py
# Shared fixtures and pytest configuration.
# Add project root to sys.path so `src.*` imports resolve when running pytest
# from the repo root: `pytest tests/`

import sys
import os

# Insert the project root (parent of this file's directory) at the front of the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
