"""
server/app.py — OpenEnv server entry point
Required by openenv validate for multi-mode deployment.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main FastAPI app
from app import app  # noqa: F401


def main():
    """Entry point for openenv serve command."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
