import pandas as pd
import sys
import os
from sqlalchemy.types import String
from datetime import datetime
from dateutil.relativedelta import relativedelta # 날짜 계산용

# 중앙 설정 파일(engine) 임포트
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from app.core.config import engine

# 특정 시군구 코드 수집을 위한 배열
TARGET_SGG_CODES = [
    '28237'
    # (나중에 서울 강서구 '11500' 등을 이곳에 추가)
]

# --- 설정 ---
# 1. 원본 법정동 코드 CSV 파일 경로
CSV_PATH = os.path.join(project_root, 'data', '국토교통부_법정동코드_20250805.csv')

TABLE_SGG = "meta_sgg_codes"  # 1. 시군구 단위 테이블 (매매, 전월세용)
TABLE_BJDONG = "meta_bjdong_codes" # 2. 법정동 단위 테이블 (건축물대장용)

# 수집 시작 날짜를 '201501'이 아닌, '현재 달의 다음 달'로 설정
# 예: 2025년 10월 -> '202511'로 설정.
# 'fetch' 스크립트가 이 값을 보고 '202510'부터 수집을 시작합니다.
NOW_MONTH = datetime.now()
DEFAULT_START_DATE = NOW_MONTH.strftime('%Y%m') # 예: '202511'

# db 최신순에서 오래된 순으로 저장
def setup_region_database():
    print(f"--- 1. 원본 법정동 코드 CSV 파일 로드 ---")
    print(f"파일 경로: {CSV_PATH}")

    try:
        # 'df' 변수 생성
        df = pd.read_csv(CSV_PATH, sep=',', encoding='cp949', dtype=str)
    except Exception as e:
        print(f"CSV 로드 실패: {e}")
        return

    print("--- 2. 데이터 정제 및 가공 ---")

    # 1. 원본 컬럼명 변경
    df = df.rename(columns={
        '법정동코드': 'code',
        '법정동명': 'name',
        '폐지여부': 'status'
    })

    # 2. 현재 사용 중인 코드('존재')만 필터링
    df_active = df[df['status'] == '존재'].copy()

    # 3. [신규] '시군구'와 '법정동' 코드를 분리 생성
    df_active['sgg_code'] = df_active['code'].str.slice(0, 5)
    df_active['bjdong_code'] = df_active['code'].str.slice(5, 10)
    df_active['bjdong_name'] = df_active['name']  # (법정동 이름)

    # --- 3-A. [시군구] 테이블 데이터 생성 (매매, 전월세용) ---
    print(f"--- 3. 목표 지역 필터링: {TARGET_SGG_CODES} ---")
    # 특정 시군구 코드만 수집하도록 필터링
    df_active = df_active[df_active['sgg_code'].isin(TARGET_SGG_CODES)].copy()

    # 1. '시군구' 레벨 코드(e.g., 11000) 제외
    is_sigungu_code = ~df_active['sgg_code'].str.endswith('000')

    # 2. '시군구' 코드 기준으로 중복 제거
    df_sgg_final = df_active[is_sigungu_code][['sgg_code']].drop_duplicates().copy()
    # 3. 수집 상태 컬럼 추가
    df_sgg_final['trade_last_fetched_date'] = DEFAULT_START_DATE
    df_sgg_final['rent_last_fetched_date'] = DEFAULT_START_DATE

    print(f"총 {len(df_sgg_final)}개의 *시군구* 코드(e.g., 11110)를 추출했습니다.")

    # --- 3-B. [법정동] 테이블 데이터 생성 (건축물대장용) ---

    # 1. '읍면동' 레벨 코드만 필터링 (시/도 '...00000000' 및 시/군/구 '...00000' 제외)
    is_dong_level = ~df_active['code'].str.endswith('00000')
    df_bjdong_final = df_active[is_dong_level][['sgg_code', 'bjdong_code', 'bjdong_name']].drop_duplicates().copy()
    # 2. 수집 상태 컬럼 추가
    df_bjdong_final['ledger_last_fetched_date'] = DEFAULT_START_DATE
    print(f"총 {len(df_bjdong_final)}개의 *법정동* 코드(e.g., 11680-10300)를 추출했습니다.")

    # --- 4. DB에 두 개의 테이블 저장 ---
    try:
        print(f"--- 4-A. DB 테이블 '{TABLE_SGG}' (시군구) 저장 중 ---")
        df_sgg_final.to_sql(
            TABLE_SGG,
            con=engine,
            if_exists='replace',
            index=False,
            dtype={
                'sgg_code': String,
                'trade_last_fetched_date': String,
                'rent_last_fetched_date': String
            }
        )
        print(f"성공: '{TABLE_SGG}' 테이블 생성이 완료되었습니다.")

        print(f"--- 4-B. DB 테이블 '{TABLE_BJDONG}' (법정동) 저장 중 ---")
        df_bjdong_final.to_sql(
            TABLE_BJDONG,
            con=engine,
            if_exists='replace',
            index=False,
            dtype={
                'sgg_code': String,
                'bjdong_code': String,
                'bjdong_name': String,
                'ledger_last_fetched_date': String
            }
        )
        print(f"성공: '{TABLE_BJDONG}' 테이블 생성이 완료되었습니다.")

    except Exception as e:
        print(f"[오류] DB 저장 실패: {e}")


if __name__ == "__main__":
    setup_region_database()