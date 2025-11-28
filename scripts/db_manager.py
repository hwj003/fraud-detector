import sqlite3
import os

# 1. DB 파일 경로 설정 (이 파일 기준으로 상위 폴더에 저장)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'local_fraud_db.sqlite'))

def get_connection():
    """DB 연결 객체를 반환하는 헬퍼 함수"""
    return sqlite3.connect(DB_PATH)

def init_db():
    """
    모든 테이블 초기화
    1. 작업 로그 (중복 방지)
    2. 집합건축물대장 전유부 (아파트, 빌라, 오피스텔 - 호수별 소유)
    3. 집합건축물대장 표제부
    """
    print(f"[DB Manager] 초기화 및 테이블 점검 시작: {DB_PATH}")

    conn = get_connection()
    cur = conn.cursor()

    # !주의: 테이블 초기화
    # cur.execute("DROP TABLE IF EXISTS api_job_log")
    # cur.execute("DROP TABLE IF EXISTS public_price_history")
    # cur.execute("DROP TABLE IF EXISTS building_info")
    # cur.execute("DROP TABLE IF EXISTS building_title_info")

    """
    테이블 명: building_info
    테이블 구조
    unique_number: 고유번호(2823710100...) - API 연동시 Key
    building_id_code: 건물ID(222004...)
    
    road_address: 도로명 주소
    lot_address: 지번주소
    detail_address: 상세주소 (101동 302호)
    
    exclusive_area: 전용면적
    main_use: 주용도
    structure_type: 구조(철근콘크리트 등)
    
    owner_name: 소유자명
    ownership_changed_date: 소유권 변동일 (최근 변경 시 위험 경고용)
    ownership_cause: 변동원인(매매/증여/신탁 등 - 신탁일 경우 위험)
    is_violating_building: 위반건축물 여부 (Y/N)
    """
    cur.execute('''
        CREATE TABLE IF NOT EXISTS building_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

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
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    # ---------------------------------------------------------
    # 2. updated_at 자동 갱신을 위한 Trigger 생성
    # ---------------------------------------------------------
    cur.execute('''
        CREATE TRIGGER IF NOT EXISTS update_building_info_modtime 
        AFTER UPDATE ON building_info
        BEGIN
            UPDATE building_info 
            SET updated_at = CURRENT_TIMESTAMP 
            WHERE id = OLD.id;
        END;
    ''')

    """
    테이블 명: public_price_history 
    테이블 구조
    building_info_id: FK
    base_date: 기준일
    price:  공시가격 (원 단위)
    created_at: 생성일
    """
    conn.execute('''
        CREATE TABLE IF NOT EXISTS public_price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        building_info_id INTEGER NOT NULL,
        base_date DATE NOT NULL,
        price DECIMAL(15, 0) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (building_info_id) REFERENCES building_info(id) ON DELETE CASCADE
    );
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS api_job_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_address VARCHAR(255) UNIQUE, -- 중복 수집 방지
            job_type VARCHAR(20) DEFAULT 'EXCLUSIVE', -- 작업 유형 ('EXCLUSIVE', 'TITLE')
            status VARCHAR(50),                 -- 작업 상태 (DETAIL_SAVED 등)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS building_title_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- 1. 식별 정보
            unique_number VARCHAR(50) NOT NULL UNIQUE, -- 고유번호 (PK)
            sigungu_code VARCHAR(10),
            bjdong_code VARCHAR(10),
            bunji VARCHAR(20),
            
            -- 2. 주소 및 건물명
            road_address VARCHAR(255),
            detail_address VARCHAR(100),               -- 건물명 (예: 광일아파트)
            dong_name VARCHAR(50),                     -- 동 명칭 (예: 1동, A동)
            
            -- 3. 건물 스펙
            main_use VARCHAR(100),                     -- 주용도 (아파트)
            structure_type VARCHAR(100),               -- 주구조 (철근콘크리트조 - 내구연한 판단용)
            total_floor_area DECIMAL(15, 2),           -- 연면적
            
            household_cnt INTEGER DEFAULT 0,           -- 총 세대수 (55세대 - 나홀로 아파트 여부 판단)
            grnd_flr_cnt INTEGER DEFAULT 0,            -- 지상 층수 (5층)
            und_flr_cnt INTEGER DEFAULT 0,             -- 지하 층수 (1층)
            
            -- 4. 편의 시설 (삶의 질 & 가격 영향)
            parking_cnt INTEGER DEFAULT 0,             -- 주차대수 (자주식+기계식 합산)
            elevator_cnt INTEGER DEFAULT 0,            -- 승강기대수 (승용+비상용)
            
            -- 5. 리스크 및 가치 지표
            use_apr_day DATE,                          -- 사용승인일 (1985-01-15 -> 재건축 가능성/노후도)
            is_violating CHAR(1) DEFAULT 'N',          -- 위반건축물 여부 (노란 딱지)
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()
    print("[DB Manager] 테이블 점검 완료")


# 테스트용 실행 코드 (이 파일을 직접 실행했을 때만 동작)
if __name__ == "__main__":
    init_db()