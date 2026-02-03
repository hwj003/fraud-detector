import os
from functools import lru_cache
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from contextlib import contextmanager
from pydantic_settings import BaseSettings
from app.core.exceptions import DatabaseConnectionError
from dotenv import load_dotenv

class Settings(BaseSettings):
    """환경 설정 - .env 파일에서 로드"""
    APP_ENV: str = "local"

    # MySQL 설정
    DB_USER: str = "root"
    DB_PASSWORD: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: str = "3306"
    DB_NAME: str = "fraud_db"

    # 연결 타임아웃 설정 (초)
    DB_CONNECT_TIMEOUT: int = 5
    DB_READ_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤"""
    return Settings()


# === Engine 관리 ===
_engine: Engine | None = None
_db_available: bool | None = None

def get_engine() -> Engine:
    """DB 엔진 싱글톤 (Lazy 초기화)"""
    global _engine

    if _engine is not None:
        return _engine

    settings = get_settings()

    db_url = (
        f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        f"?charset=utf8mb4"
        f"&connect_timeout={settings.DB_CONNECT_TIMEOUT}"
        f"&read_timeout={settings.DB_READ_TIMEOUT}"
    )

    _engine = create_engine(
        db_url,
        echo=(settings.APP_ENV == "local"),  # 로컬에서만 SQL 로그
        pool_recycle=3600,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping = True, # 연결 상태 사전 확인
        pool_timeout=settings.DB_CONNECT_TIMEOUT, # 풀에서 연결 대기 타임아웃
    )

    return _engine


def check_db_connection() -> bool:
    """
    데이터베이스 연결 상태 확인

    Returns:
        연결 가능 여부
    """
    global _db_available

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _db_available = True
        return True
    except (OperationalError, SQLAlchemyError) as e:
        _db_available = False
        return False


def is_db_available() -> bool:
    """
    DB 가용성 캐시 확인 (빠른 체크)

    최초 호출 시에만 실제 연결 테스트
    """
    global _db_available

    if _db_available is None:
        return check_db_connection()
    return _db_available


def reset_db_availability():
    """DB 가용성 캐시 리셋 (재연결 시도용)"""
    global _db_available
    _db_available = None


def get_connection_with_check():
    """
    연결 상태 확인 후 연결 반환

    Raises:
        DatabaseConnectionError: 연결 실패 시
    """
    try:
        engine = get_engine()
        conn = engine.connect()
        return conn
    except OperationalError as e:
        reset_db_availability()
        raise DatabaseConnectionError(
            "MySQL 서버에 연결할 수 없습니다",
            original_error=e
        )

# === Session 관리 ===
_SessionLocal = None


def get_session_factory():
    """세션 팩토리 싱글톤"""
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )

    return _SessionLocal


@contextmanager
def get_db_session():
    """
    컨텍스트 매니저로 세션 관리

    Raises:
        DatabaseConnectionError: 연결 실패 시

    사용법:
        with get_db_session() as session:
            session.execute(...)
    """
    # 연결 가능 여부 먼저 확인
    if not is_db_available():
        if not check_db_connection():  # 재확인
            raise DatabaseConnectionError("데이터베이스에 연결할 수 없습니다")

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except OperationalError as e:
        db.rollback()
        reset_db_availability()
        raise DatabaseConnectionError(
            "데이터베이스 연결이 끊어졌습니다",
            original_error=e
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    """FastAPI Dependency Injection용 제너레이터"""
    # 연결 가능 여부 먼저 확인
    if not is_db_available():
        if not check_db_connection():
            raise DatabaseConnectionError("데이터베이스에 연결할 수 없습니다")

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except OperationalError as e:
        db.rollback()
        reset_db_availability()
        raise DatabaseConnectionError(
            "데이터베이스 연결이 끊어졌습니다",
            original_error=e
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# === 하위 호환성 ===
def __getattr__(name: str):
    """기존 코드 호환: from app.core.config import engine"""
    if name == "engine":
        return get_engine()
    if name == "load_dotenv":
        from dotenv import load_dotenv
        return load_dotenv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")