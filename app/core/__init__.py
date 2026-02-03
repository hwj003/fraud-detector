"""
Core 모듈

설정 및 데이터베이스 관련 기능 제공
"""
# 설정
from app.core.config import get_settings, Settings

# 데이터베이스
from app.core.database import (
    get_engine,
    get_connection,
    get_connection_with_check,
    get_db_session,
    get_db,
    check_db_connection,
    is_db_available,
    reset_db_availability,
    init_db,
)

# 예외
from app.core.exceptions import (
    DatabaseConnectionError,
    DatabaseOperationError,
    ServiceUnavailableError,
)

__all__ = [
    # config
    "get_settings",
    "Settings",
    # database
    "get_engine",
    "get_connection",
    "get_connection_with_check",
    "get_db_session",
    "get_db",
    "check_db_connection",
    "is_db_available",
    "reset_db_availability",
    "init_db",
    # exceptions
    "DatabaseConnectionError",
    "DatabaseOperationError",
    "ServiceUnavailableError",
]
