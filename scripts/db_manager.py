"""
[DEPRECATED] 이 파일은 하위 호환성을 위해 유지됩니다.

새 코드에서는 다음을 사용하세요:
    from app.core.database import get_engine, get_connection, init_db
    from app.core.config import get_settings

또는:
    from app.core import get_engine, get_connection, init_db, get_settings
"""
import warnings

# 경고 메시지 (개발 중에만 표시)
warnings.warn(
    "scripts.db_manager는 deprecated입니다. "
    "app.core.database를 사용하세요.",
    DeprecationWarning,
    stacklevel=2
)

# 기존 import 호환성 유지
from app.core.database import (
    get_engine,
    get_connection,
    get_db_session,
    get_db,
    init_db,
)
from app.core.config import get_settings

# 기존 코드에서 engine을 직접 import하는 경우 지원
# from scripts.db_manager import engine
def __getattr__(name: str):
    if name == "engine":
        return get_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "get_engine",
    "get_connection",
    "get_db_session",
    "get_db",
    "get_settings",
    "init_db",
]
