
from sqlalchemy import  text

from app.core.config import (
    get_engine,
    get_db_session,
    get_db,
    get_settings
)

# 외부에서 사용할 수 있도록 re-export
__all__ = ['get_engine', 'get_db_session', 'get_db', 'get_connection', 'init_db']

def get_connection():
    """
    Raw connection 반환 (레거시 호환용)
    가능하면 get_db_session() 사용을 권장합니다.
    """
    return get_engine().connect()

def init_db():
    """MySQL 테이블 초기화"""
    settings = get_settings()
    print(f"[DB Manager] MySQL 테이블 초기화 시작 ({settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME})")

    engine = get_engine()

    # SQLAlchemy 엔진을 통해 직접 연결
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
        # 2. building_title_info (건물 표제부 정보 - 주차장, 승강기 등)
        # -----------------------------------------------------
        conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS building_title_info (
                        id INT AUTO_INCREMENT PRIMARY KEY,

                        -- 1. 식별 정보
                        unique_number VARCHAR(50) NOT NULL UNIQUE,
                        sigungu_code VARCHAR(10),
                        bjdong_code VARCHAR(10),
                        bunji VARCHAR(20),

                        -- 2. 주소 및 건물명
                        road_address VARCHAR(255),
                        detail_address VARCHAR(100),
                        dong_name VARCHAR(50),

                        -- 3. 건물 스펙
                        main_use VARCHAR(100),
                        structure_type VARCHAR(100),
                        total_floor_area DECIMAL(15, 2),

                        household_cnt INT DEFAULT 0,
                        grnd_flr_cnt INT DEFAULT 0,
                        und_flr_cnt INT DEFAULT 0,

                        -- 4. 편의 시설
                        parking_cnt INT DEFAULT 0,
                        elevator_cnt INT DEFAULT 0,

                        -- 5. 리스크 및 가치 지표
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
        # 4. api_price_log (API 호출 이력 - 실거래/전세 데이터)
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

        # [데이터 초기화] meta_bjdong_codes 테이블이 존재한다고 가정
        # MySQL에서는 INSERT OR IGNORE 대신 INSERT IGNORE 사용
        # (주의: meta_bjdong_codes 테이블이 먼저 생성되어 있어야 실행됨)
        try:
            conn.execute(text("""
                        INSERT IGNORE INTO job_sgg_history (sgg_code, status)
                        SELECT DISTINCT sgg_code, 'READY'
                        FROM meta_bjdong_codes
                        WHERE sgg_code IS NOT NULL;
                    """))
        except Exception as e:
            print(f"[Warning] job_sgg_history 초기화 실패 (meta_bjdong_codes 테이블 없음?): {e}")

        # -----------------------------------------------------
        # 6. regional_stats (지역별 통계 요약 - 성능 최적화용)
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
        # 7. risk_analysis_result (전세사기 위험도 분석 결과)
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
        # [수정] official_price_raw (공시지가 원천 데이터)
        # SQLite 문법(AUTOINCREMENT)을 MySQL 문법(AUTO_INCREMENT)으로 변경
        # -----------------------------------------------------
        conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS official_price_raw (
                        id INT AUTO_INCREMENT PRIMARY KEY,

                        -- 1. 핵심 식별자 (검색용)
                        pnu VARCHAR(19) NOT NULL,       -- PNU
                        sigungu_code VARCHAR(5),
                        bjdong_code VARCHAR(5),

                        -- 2. 상세 주소
                        dong_name VARCHAR(50),
                        ho_name VARCHAR(50),

                        -- 3. 데이터
                        price DECIMAL(15, 0),
                        exclusive_area DECIMAL(10, 2),
                        base_year VARCHAR(4),

                        -- 4. 기타
                        complex_name VARCHAR(100),
                        road_address VARCHAR(255),

                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                        -- [중요] 조회 속도를 위해 PNU에 인덱스 추가
                        INDEX idx_pnu (pnu)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """))

        # -----------------------------------------------------
        # 8. raw_rent (국토부 전월세 실거래가 원천 데이터)
        # -----------------------------------------------------
        conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS raw_rent (
                        id INT AUTO_INCREMENT PRIMARY KEY,

                        시군구 VARCHAR(50),
                        법정동 VARCHAR(50),
                        본번 VARCHAR(20),
                        부번 VARCHAR(20),

                        보증금 VARCHAR(50),    -- '50,000' 같은 문자열 처리를 위해 VARCHAR 권장
                        월세 VARCHAR(50),

                        계약일 VARCHAR(20),    -- '20240101' 형태
                        계약유형 VARCHAR(20),  -- 신규/갱신
                        건물유형 VARCHAR(20),  -- 아파트/연립다세대 등

                        층 VARCHAR(20),
                        전용면적 VARCHAR(30),
                        건물명 VARCHAR(100),
                        건축년도 VARCHAR(10),

                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                        -- 조회 성능 향상을 위한 인덱스
                        INDEX idx_rent_sigungu (시군구),
                        INDEX idx_rent_date (계약일)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """))

        # -----------------------------------------------------
        # 9. raw_trade (국토부 매매 실거래가 원천 데이터)
        # -----------------------------------------------------
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_trade (
                id INT AUTO_INCREMENT PRIMARY KEY,
    
                시군구 VARCHAR(50),
                법정동 VARCHAR(50),
                본번 VARCHAR(20),
                부번 VARCHAR(20),
    
                거래금액 VARCHAR(50),  -- 쉼표 포함 가능성 고려하여 문자열 저장
                계약일 VARCHAR(20),
    
                전용면적 VARCHAR(30),
                층 VARCHAR(20),
                건물명 VARCHAR(100),
                건축년도 VARCHAR(10),
                건물유형 VARCHAR(20),
    
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
                -- 조회 성능 향상을 위한 인덱스
                INDEX idx_trade_sigungu (시군구),
                INDEX idx_trade_date (계약일)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS regions (
                region_code VARCHAR(10) PRIMARY KEY COMMENT '시군구 코드 (예: 11110)',
                region_name VARCHAR(50) NOT NULL COMMENT '시군구 명 (예: 서울특별시 종로구)'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))
        print("[DB Manager] 모든 MySQL 테이블 생성 완료")

# 테스트용 실행 코드 (이 파일을 직접 실행했을 때만 동작)
if __name__ == "__main__":
    init_db()