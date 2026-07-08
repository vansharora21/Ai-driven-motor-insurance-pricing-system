from __future__ import annotations

import uvicorn

from src.api import app

if __name__ == "__main__":
    uvicorn.run(
        "scripts.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
