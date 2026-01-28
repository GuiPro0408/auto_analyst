"""Chainlit data layer for chat persistence.

Uses SQLite for local storage of chat threads and messages.
"""

import os
from pathlib import Path
from typing import Any, Dict, Union

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer  # type: ignore[import-not-found]
from chainlit.data.storage_clients.base import BaseStorageClient  # type: ignore[import-not-found]

# Ensure data directory exists
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "chainlit_chats.db"
UPLOADS_DIR = DATA_DIR / "chainlit_uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


class LocalStorageClient(BaseStorageClient):
    """Simple local file system storage client for Chainlit."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def upload_file(
        self,
        object_key: str,
        data: Union[bytes, str],
        mime: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """Upload a file to local storage."""
        file_path = self.base_path / object_key
        file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "wb" if isinstance(data, bytes) else "w"
        with open(file_path, mode) as f:
            f.write(data)

        return {
            "object_key": object_key,
            "url": f"file://{file_path}",
        }

    async def delete_file(self, object_key: str) -> bool:
        """Delete a file from local storage."""
        file_path = self.base_path / object_key
        if file_path.exists():
            os.remove(file_path)
            return True
        return False

    async def get_read_url(self, object_key: str) -> str:
        """Get a URL for reading a file."""
        file_path = self.base_path / object_key
        return f"file://{file_path}"

    async def close(self) -> None:
        """Close the storage client (no-op for local storage)."""
        pass


def get_data_layer() -> SQLAlchemyDataLayer:
    """Create a SQLite-backed data layer for chat persistence."""
    storage_client = LocalStorageClient(str(UPLOADS_DIR))

    return SQLAlchemyDataLayer(
        conninfo=f"sqlite+aiosqlite:///{DB_PATH}",
        storage_provider=storage_client,
        show_logger=False,
    )
