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
    2. 일반건축물대장 (다가구, 상가주택 등 - 1인 소유)
    3. 집합건축물대장 전유부 (아파트, 빌라, 오피스텔 - 호수별 소유)
    """
    print(f"[DB Manager] 초기화 및 테이블 점검 시작: {DB_PATH}")

    conn = get_connection()
    cur = conn.cursor()

    # ---------------------------------------------------------
    # 1. 작업 로그 테이블 (API 호출 상태 기록)
    # ---------------------------------------------------------
    cur.execute('''
            CREATE TABLE IF NOT EXISTS api_job_log (
                search_address TEXT PRIMARY KEY,  -- 검색한 주소
                status TEXT,                      -- 성공/실패/보류 상태
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    # ---------------------------------------------------------
    # 2. 일반건축물 테이블 (General Buildings)
    # 대상: 다가구, 단독주택, 상가 (건물주가 1명인 경우)
    # ---------------------------------------------------------
    cur.execute('''
            CREATE TABLE IF NOT EXISTS general_buildings (
                unique_no TEXT PRIMARY KEY,       -- 고유번호
                road_addr TEXT,                   -- 도로명주소
                lot_addr TEXT,                    -- 지번주소
                owner_name TEXT,                  -- 소유자명 (건물 전체 주인)
                main_usage TEXT,                  -- 주용도
                total_area REAL,                  -- 연면적
                violation_details TEXT,           -- 위반건축물 내역
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

    # ---------------------------------------------------------
    # 3. 집합건축물 전유부 테이블 (Collective Units)
    # 대상: 아파트, 빌라, 오피스텔 (호수별로 주인이 다름)
    # 핵심: (고유번호 + 동 + 호)가 합쳐져야 유일한 키가 됨
    # ---------------------------------------------------------
    cur.execute('''
            CREATE TABLE IF NOT EXISTS collective_units (
                unique_no TEXT,                   -- 고유번호 (단지 식별용)
                dong_nm TEXT,                     -- 동 (예: 101동)
                ho_nm TEXT,                       -- 호 (예: 502호)
                road_addr TEXT,                   -- 도로명주소
                owner_name TEXT,                  -- 소유자명 (해당 호수의 주인)
                exclusive_area REAL,              -- 전용면적 (가장 중요)
                violation_details TEXT,           -- 위반 내역
                total_area REAL,                  -- 공급면적(참고용)
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- [중요] 복합 기본키 설정 (이 3개가 같으면 중복 데이터로 간주)
                PRIMARY KEY (unique_no, dong_nm, ho_nm)
            )
        ''')

    conn.commit()
    conn.close()
    print("[DB Manager] 테이블 점검 완료")


# 테스트용 실행 코드 (이 파일을 직접 실행했을 때만 동작)
if __name__ == "__main__":
    init_db()