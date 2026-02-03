import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
import os

# ---------------------------------------------------------
# 1. 설정: 경로 및 접속 정보
# ---------------------------------------------------------
# (1) SQLite 파일 경로 (기존 데이터)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'local_fraud_db.sqlite'))

# (2) MySQL 접속 정보 (Docker)
DB_HOST = "localhost"
DB_PORT = "3306"
DB_USER = "root"
DB_PASSWORD = "2345"
DB_NAME = "fraud_db"

MYSQL_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

def migrate():
    print(f"[Migration] 시작: SQLite({SQLITE_PATH}) -> MySQL({DB_NAME})")

    # 1. SQLite 연결
    if not os.path.exists(SQLITE_PATH):
        print(f"[Error] SQLite 파일을 찾을 수 없습니다: {SQLITE_PATH}")
        return

    sqlite_conn = sqlite3.connect(SQLITE_PATH)

    # 2. MySQL 엔진 생성
    mysql_engine = create_engine(MYSQL_URL)

    # 3. 옮길 테이블 목록 정의
    # (주의: 이미 MySQL에 init_db()로 테이블을 만들어 둔 상태여야 합니다)
    target_tables = [
        "meta_bjdong_codes"
    ]

    try:
        for table in target_tables:
            print(f"   Copying table: {table}...", end=" ")

            # (1) SQLite에서 데이터 읽기 (Pandas DataFrame)
            try:
                df = pd.read_sql(f"SELECT * FROM {table}", sqlite_conn)
            except Exception:
                print(f"[Skip] SQLite에 {table} 테이블이 없어서 건너뜁니다.")
                continue

            if df.empty:
                print("[Skip] 데이터가 없어서 건너뜁니다.")
                continue

            # (2) 데이터 전처리 (SQLite -> MySQL 호환성 맞추기)
            # SQLite의 'id' 컬럼이 있다면 제거 (MySQL의 AUTO_INCREMENT가 처리하도록)
            if 'id' in df.columns:
                df = df.drop(columns=['id'])

            # 날짜 포맷 등이 문제될 경우 여기서 변환 가능
            # 예: df['created_at'] = pd.to_datetime(df['created_at'])

            # (3) MySQL에 데이터 밀어넣기
            # if_exists='append': 기존 테이블 구조 유지하고 데이터만 추가
            # index=False: DataFrame의 index(0,1,2...)는 저장 안 함
            df.to_sql(name=table, con=mysql_engine, if_exists='append', index=False)

            print(f"Done! ({len(df)} rows)")

        print("[Migration] 모든 작업이 완료되었습니다.")

    except Exception as e:
        print(f"\n[Error] 마이그레이션 중 오류 발생: {e}")

    finally:
        sqlite_conn.close()

if __name__ == "__main__":
    migrate()