import pytest
from app.core.config import reset_db_availability

@pytest.fixture(autouse=True)
def reset_db_state():
    """각 테스트 전후로 DB 상태 초기화"""
    reset_db_availability()
    yield
    reset_db_availability()