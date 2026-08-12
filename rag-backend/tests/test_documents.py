import pytest
import pytest_asyncio
import uuid
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import engine, AsyncSessionLocal
from app.database.repositories import create_user, create_document, get_document_by_id, soft_delete_document

@pytest_asyncio.fixture
async def db():
    async with AsyncSessionLocal() as session:
        yield session

@pytest_asyncio.fixture
async def test_user(db: AsyncSession):
    user = await create_user(db, f"test_{uuid.uuid4()}", "hash")
    await db.commit()
    return user

@pytest.mark.asyncio
async def test_first_upload(db: AsyncSession, test_user):
    doc = await create_document(db, test_user.id, "report.pdf")
    await db.commit()
    assert doc.original_filename == "report.pdf"
    assert doc.display_name == "report.pdf"
    assert doc.duplicate_index == 0

@pytest.mark.asyncio
async def test_second_upload(db: AsyncSession, test_user):
    doc1 = await create_document(db, test_user.id, "report.pdf")
    doc2 = await create_document(db, test_user.id, "report.pdf")
    await db.commit()
    
    assert doc2.original_filename == "report.pdf"
    assert doc2.display_name == "report (2).pdf"
    assert doc2.duplicate_index == 1

@pytest.mark.asyncio
async def test_third_upload(db: AsyncSession, test_user):
    await create_document(db, test_user.id, "data.csv")
    await create_document(db, test_user.id, "data.csv")
    doc3 = await create_document(db, test_user.id, "data.csv")
    await db.commit()
    
    assert doc3.original_filename == "data.csv"
    assert doc3.display_name == "data (3).csv"
    assert doc3.duplicate_index == 2

@pytest.mark.asyncio
async def test_deletion_followed_by_upload(db: AsyncSession, test_user):
    doc1 = await create_document(db, test_user.id, "invoice.pdf")
    await db.commit()
    await soft_delete_document(db, doc1.id, test_user.id)
    await db.commit()
    
    doc2 = await create_document(db, test_user.id, "invoice.pdf")
    await db.commit()
    assert doc2.duplicate_index == 1
    assert doc2.display_name == "invoice (2).pdf"

@pytest.mark.asyncio
async def test_different_users_uploading_same_filename(db: AsyncSession):
    # This one needs two users
    user1 = await create_user(db, f"test_{uuid.uuid4()}", "hash")
    user2 = await create_user(db, f"test_{uuid.uuid4()}", "hash")
    await db.commit()
    
    doc1 = await create_document(db, user1.id, "budget.xlsx")
    doc2 = await create_document(db, user2.id, "budget.xlsx")
    await db.commit()
    
    assert doc1.duplicate_index == 0
    assert doc1.display_name == "budget.xlsx"
    assert doc2.duplicate_index == 0
    assert doc2.display_name == "budget.xlsx"

@pytest.mark.asyncio
async def test_unicode_filenames(db: AsyncSession, test_user):
    doc1 = await create_document(db, test_user.id, "こんにちは.txt")
    doc2 = await create_document(db, test_user.id, "こんにちは.txt")
    await db.commit()
    
    assert doc1.display_name == "こんにちは.txt"
    assert doc2.display_name == "こんにちは (2).txt"

@pytest.mark.asyncio
async def test_filenames_with_multiple_dots(db: AsyncSession, test_user):
    doc1 = await create_document(db, test_user.id, "archive.tar.gz")
    doc2 = await create_document(db, test_user.id, "archive.tar.gz")
    await db.commit()
    
    assert doc1.display_name == "archive.tar.gz"
    assert doc2.display_name == "archive (2).tar.gz"

@pytest.mark.asyncio
async def test_filenames_with_no_extension(db: AsyncSession, test_user):
    doc1 = await create_document(db, test_user.id, "README")
    doc2 = await create_document(db, test_user.id, "README")
    await db.commit()
    
    assert doc1.display_name == "README"
    assert doc2.display_name == "README (2)"
