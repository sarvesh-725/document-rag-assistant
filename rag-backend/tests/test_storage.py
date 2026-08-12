import pytest
import uuid
import os
from app.services.storage import LocalStorageService

@pytest.fixture
def storage_service(tmp_path):
    # tmp_path is a built-in pytest fixture for temporary directories
    return LocalStorageService(str(tmp_path))

@pytest.mark.asyncio
async def test_storage_save_and_read(storage_service):
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    version_id = uuid.uuid4()
    
    storage_key = f"documents/{user_id}/{doc_id}/{version_id}/original"
    content = b"test content"
    
    # Save
    await storage_service.save(storage_key, content, user_id)
    
    # Exists
    assert await storage_service.exists(storage_key, user_id) is True
    
    # Read
    read_content = await storage_service.read(storage_key, user_id)
    assert read_content == content

@pytest.mark.asyncio
async def test_storage_delete(storage_service):
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    version_id = uuid.uuid4()
    
    storage_key = f"documents/{user_id}/{doc_id}/{version_id}/original"
    content = b"test content"
    
    await storage_service.save(storage_key, content, user_id)
    assert await storage_service.exists(storage_key, user_id) is True
    
    # Delete
    deleted = await storage_service.delete(storage_key, user_id)
    assert deleted is True
    
    # Exists should be False
    assert await storage_service.exists(storage_key, user_id) is False
    
    # Deleting again should return False
    assert await storage_service.delete(storage_key, user_id) is False

@pytest.mark.asyncio
async def test_storage_security_wrong_user(storage_service):
    user_id = uuid.uuid4()
    wrong_user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    version_id = uuid.uuid4()
    
    storage_key = f"documents/{user_id}/{doc_id}/{version_id}/original"
    content = b"test content"
    
    await storage_service.save(storage_key, content, user_id)
    
    # Trying to read with wrong user_id should raise PermissionError
    with pytest.raises(PermissionError):
        await storage_service.read(storage_key, wrong_user_id)
        
    # Exists with wrong user_id should safely return False
    assert await storage_service.exists(storage_key, wrong_user_id) is False

@pytest.mark.asyncio
async def test_storage_path_traversal(storage_service):
    user_id = uuid.uuid4()
    # Try to break out of the directory
    storage_key = f"documents/{user_id}/../../../etc/passwd"
    
    # Should raise ValueError for traversal
    with pytest.raises(ValueError):
        await storage_service.save(storage_key, b"hack", user_id)
        
    with pytest.raises(ValueError):
        await storage_service.read(storage_key, user_id)

@pytest.mark.asyncio
async def test_storage_invalid_format(storage_service):
    user_id = uuid.uuid4()
    # Path doesn't start with documents/{user_id}/
    storage_key = "some_other_path/original"
    
    with pytest.raises(PermissionError):
        await storage_service.save(storage_key, b"test", user_id)
