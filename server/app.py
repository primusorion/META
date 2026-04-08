from __future__ import annotations

import os

import uvicorn

from app.main import app


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
