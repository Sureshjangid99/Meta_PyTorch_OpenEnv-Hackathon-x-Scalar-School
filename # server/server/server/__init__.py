# server/app.py — entry point required by openenv validate
# Imports and re-exports the main FastAPI app
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

__all__ = ["app"]
