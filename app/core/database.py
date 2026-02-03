"""
데이터베이스 연결 및 관리

담당 기능:
- DB 엔진 생성 및 관리
- 세션 팩토리
- 연결 상태 확인
- 테이블 초기화 (init_db)
"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.core.config import get_settings
from app.core.exceptions import DatabaseConnectionError


# =============================================================================
# Engine 관리 (싱글톤)
# =============================================================================
_engine: Engine | None = None
_db_available: bool | None = None
_SessionLocal = None


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
        pool_pre_ping=True,  # 연결 상태 사전 확인
        pool_timeout=settings.DB_CONNECT_TIMEOUT,
    )

    return _engine


def get_connection() -> Connection:
    """
    Raw connection 반환

    주의: 사용 후 반드시 close() 호출 필요
    가능하면 get_db_session() 사용 권장

    Returns:
        SQLAlchemy Connection 객체
    """
    return get_engine().connect()


def get_connection_with_check() -> Connection:
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


# =============================================================================
# 연결 상태 관리
# =============================================================================
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
    except (OperationalError, SQLAlchemyError):
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


# =============================================================================
# Session 관리
# =============================================================================
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
def get_db_session() -> Generator[Session, None, None]:
    """
    컨텍스트 매니저로 세션 관리

    Raises:
        DatabaseConnectionError: 연결 실패 시

    사용법:
        with get_db_session() as session:
            session.execute(...)
    """
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


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency Injection용 제너레이터"""
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


# =============================================================================
# 테이블 초기화
# =============================================================================
def init_db():
    """MySQL 테이블 초기화"""
    settings = get_settings()
    print(f"[Database] 테이블 초기화 시작 ({settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME})")

    engine = get_engine()

    with engine.begin() as conn:
        # -----------------------------------------------------
        # 1. building_info (건물 기본 정보 - 전유부 위주)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS building_info (
                id INT AUTO_INCREMENT PRIMARY KEY,
                unique_number VARCHAR(50) NOT NULL UNIQUE,
                building_id_code VARCHAR(50),

                road_address VARCHAR(255) NOT NULL,
                lot_address VARCHAR(255),
                detail_address VARCHAR(100),

                exclusive_area DECIMAL(10, 2) NOT NULL,
                main_use VARCHAR(50) NOT NULL,
                structure_type VARCHAR(50),

                owner_name VARCHAR(100),
                ownership_changed_date DATE,
                ownership_cause VARCHAR(50),
                is_violating_building CHAR(1) DEFAULT 'N',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 2. building_title_info (건물 표제부 정보)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS building_title_info (
                id INT AUTO_INCREMENT PRIMARY KEY,

                unique_number VARCHAR(50) NOT NULL UNIQUE,
                sigungu_code VARCHAR(10),
                bjdong_code VARCHAR(10),
                bunji VARCHAR(20),

                road_address VARCHAR(255),
                detail_address VARCHAR(100),
                dong_name VARCHAR(50),

                main_use VARCHAR(100),
                structure_type VARCHAR(100),
                total_floor_area DECIMAL(15, 2),

                household_cnt INT DEFAULT 0,
                grnd_flr_cnt INT DEFAULT 0,
                und_flr_cnt INT DEFAULT 0,

                parking_cnt INT DEFAULT 0,
                elevator_cnt INT DEFAULT 0,

                use_apr_day DATE,
                is_violating CHAR(1) DEFAULT 'N',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 3. public_price_history (공시지가 이력)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public_price_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                building_info_id INT NOT NULL,
                base_date DATE NOT NULL,
                price DECIMAL(15, 0) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (building_info_id) REFERENCES building_info(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 4. api_price_log (API 호출 이력)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS api_price_log (
                sigungu_code VARCHAR(10),
                deal_ymd VARCHAR(6),
                data_type VARCHAR(10),
                collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (sigungu_code, deal_ymd, data_type)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 5. job_sgg_history (작업 상태 관리)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS job_sgg_history (
                sgg_code VARCHAR(10) PRIMARY KEY,
                status VARCHAR(20) DEFAULT 'READY',
                last_worked_at TIMESTAMP,
                message TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        try:
            conn.execute(text("""
                INSERT IGNORE INTO job_sgg_history (sgg_code, status)
                SELECT DISTINCT sgg_code, 'READY'
                FROM meta_bjdong_codes
                WHERE sgg_code IS NOT NULL;
            """))
        except Exception as e:
            print(f"[Warning] job_sgg_history 초기화 스킵: {e}")

        # -----------------------------------------------------
        # 6. regional_stats (지역별 통계 요약)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS regional_stats (
                id INT AUTO_INCREMENT PRIMARY KEY,
                region_code VARCHAR(10) NOT NULL,
                region_name VARCHAR(50),
                month VARCHAR(7) NOT NULL,
                avg_ratio DECIMAL(5, 1),
                tx_count INT,
                risk_level VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE KEY uk_region_month (region_code, month),
                INDEX idx_month (month)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 7. risk_analysis_result (위험도 분석 결과)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS risk_analysis_result (
                id INT AUTO_INCREMENT PRIMARY KEY,

                address_key VARCHAR(255),
                building_info_id INT,

                jeonse_ratio DECIMAL(5, 2),
                hug_safe_limit BIGINT,
                hug_risk_ratio DECIMAL(5, 2),
                total_risk_ratio DECIMAL(5, 2),
                estimated_loan_amount BIGINT,

                risk_level VARCHAR(20),
                risk_score INT,

                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                INDEX idx_risk_address (address_key),
                INDEX idx_risk_level (risk_level)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 8. official_price_raw (공시지가 원천 데이터)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS official_price_raw (
                id INT AUTO_INCREMENT PRIMARY KEY,

                pnu VARCHAR(19) NOT NULL,
                sigungu_code VARCHAR(5),
                bjdong_code VARCHAR(5),

                dong_name VARCHAR(50),
                ho_name VARCHAR(50),

                price DECIMAL(15, 0),
                exclusive_area DECIMAL(10, 2),
                base_year VARCHAR(4),

                complex_name VARCHAR(100),
                road_address VARCHAR(255),

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                INDEX idx_pnu (pnu)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 9. raw_rent (전월세 실거래가)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_rent (
                id INT AUTO_INCREMENT PRIMARY KEY,

                시군구 VARCHAR(50),
                법정동 VARCHAR(50),
                본번 VARCHAR(20),
                부번 VARCHAR(20),

                보증금 VARCHAR(50),
                월세 VARCHAR(50),

                계약일 VARCHAR(20),
                계약유형 VARCHAR(20),
                건물유형 VARCHAR(20),

                층 VARCHAR(20),
                전용면적 VARCHAR(30),
                건물명 VARCHAR(100),
                건축년도 VARCHAR(10),

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                INDEX idx_rent_sigungu (시군구),
                INDEX idx_rent_date (계약일)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 10. raw_trade (매매 실거래가)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_trade (
                id INT AUTO_INCREMENT PRIMARY KEY,

                시군구 VARCHAR(50),
                법정동 VARCHAR(50),
                본번 VARCHAR(20),
                부번 VARCHAR(20),

                거래금액 VARCHAR(50),
                계약일 VARCHAR(20),

                전용면적 VARCHAR(30),
                층 VARCHAR(20),
                건물명 VARCHAR(100),
                건축년도 VARCHAR(10),
                건물유형 VARCHAR(20),

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                INDEX idx_trade_sigungu (시군구),
                INDEX idx_trade_date (계약일)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        # -----------------------------------------------------
        # 11. regions (지역 정보)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS regions (
                region_code VARCHAR(10) PRIMARY KEY COMMENT '시군구 코드',
                region_name VARCHAR(50) NOT NULL COMMENT '시군구 명'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        print("[Database] 모든 테이블 생성 완료")


# =============================================================================
# 하위 호환성 (기존 import 지원)
# =============================================================================
# from app.core.database import engine 형태로 사용 가능
def __getattr__(name: str):
    if name == "engine":
        return get_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# CLI 실행
# =============================================================================
if __name__ == "__main__":
    init_db()
