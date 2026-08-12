import os
import uuid
import shutil
from typing import Protocol, BinaryIO

class StorageService(Protocol):
    async def save(self, storage_key: str, content: bytes, user_id: uuid.UUID) -> None:
        ...

    async def read(self, storage_key: str, user_id: uuid.UUID) -> bytes:
        ...

    async def delete(self, storage_key: str, user_id: uuid.UUID) -> bool:
        ...

    async def exists(self, storage_key: str, user_id: uuid.UUID) -> bool:
        ...

class LocalStorageService(StorageService):
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_absolute_path(self, storage_key: str, user_id: uuid.UUID) -> str:
        # Validate format (e.g. documents/{user_id}/{document_id}/{version_id}/original)
        if not storage_key.startswith(f"documents/{str(user_id)}/"):
            raise PermissionError(f"User {user_id} does not have access to this storage key or format is invalid.")
            
        # Ensure path traversal is blocked
        full_path = os.path.abspath(os.path.join(self.base_dir, storage_key))
        if not full_path.startswith(self.base_dir):
            raise ValueError("Path traversal detected")
            
        return full_path

    async def save(self, storage_key: str, content: bytes, user_id: uuid.UUID) -> None:
        full_path = self._get_absolute_path(storage_key, user_id)
        
        # Ensure target directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, "wb") as f:
            f.write(content)

    async def read(self, storage_key: str, user_id: uuid.UUID) -> bytes:
        full_path = self._get_absolute_path(storage_key, user_id)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Storage key {storage_key} not found.")
        
        with open(full_path, "rb") as f:
            return f.read()

    async def delete(self, storage_key: str, user_id: uuid.UUID) -> bool:
        full_path = self._get_absolute_path(storage_key, user_id)
        if not os.path.exists(full_path):
            return False
            
        os.remove(full_path)
        
        # Cleanup empty parent directories
        current_dir = os.path.dirname(full_path)
        while current_dir != self.base_dir:
            try:
                os.rmdir(current_dir)
                current_dir = os.path.dirname(current_dir)
            except OSError:
                # Directory is not empty or permission denied
                break
                
        return True

    async def exists(self, storage_key: str, user_id: uuid.UUID) -> bool:
        try:
            full_path = self._get_absolute_path(storage_key, user_id)
            return os.path.exists(full_path)
        except (PermissionError, ValueError):
            return False
