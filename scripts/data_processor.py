# fraud_detector_project/scripts/data_processor.py

import pandas as pd
import numpy as np  # inf 처리를 위해 numpy 임포트
import re
import os
import sys
from datetime import datetime

# --- [필수] 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# --- 중앙 설정 파일(engine) 임포트 ---
from app.core.config import engine


# --- 1. 헬퍼 함수 (키 생성) ---
def _create_join_key(row, keys=['시군구', '법정동', '본번', '부번']):
    """'11110-10100-0053-0078'와 같은 고유 식별 키를 생성합니다."""
    try:
        sgg = str(row[keys[0]]).strip()
        bjd = str(row[keys[1]]).strip()
        bon = str(row[keys[2]]).split('.')[0].zfill(4).strip()
        bu = str(row[keys[3]]).split('.')[0].zfill(4).strip()
        return f"{sgg}-{bjd}-{bon}-{bu}"
    except KeyError as e:
        print(f"키 생성 오류: 필요한 컬럼 {e}가 없습니다.")
        return None
    except Exception as e:
        print(f"키 생성 중 알 수 없는 오류: {e}")
        return None


# --- 2. 메인 데이터 가공 함수 ---
def load_and_engineer_features() -> pd.DataFrame:
    """
    DB의 raw 테이블 3개(rent, trade, ledger)를 JOIN하여
    가상 등기부 데이터를 포함한 모델 학습용 특성을 생성합니다.
    """

    print("--- 1. 원본 데이터 로드 중 (DB) ---")

    # [주의] 이 컬럼명들은 fetch_*.py 스크립트가 DB에 저장하는 이름과 100% 일치해야 합니다.
    SQL_RENT = """
               SELECT 시군구, \
                      법정동, \
                      본번, \
                      부번, \
                      보증금 AS RENT_PRICE, \
                      월세  AS MONTHLY_RENT, \
                      계약일 AS CONTRACT_DATE
               FROM raw_rent \
               """

    SQL_TRADE = """
                SELECT 시군구, \
                       법정동, \
                       본번, \
                       부번, \
                       `거래금액(만원)` AS TRADE_PRICE, \
                       계약일        AS TRADE_DATE
                FROM raw_trade \
                """

    SQL_LEDGER = """
                 SELECT 시군구, \
                        법정동, \
                        본번, \
                        부번, \
                        주용도     AS MAIN_PURPOSE, \
                        위반건축물여부 AS IS_ILLEGAL, \
                        사용승인일   AS USE_APR_DAY
                 FROM raw_ledger \
                 """

    try:
        print("  -> 'raw_rent' 로드 중...")
        df_rent = pd.read_sql(SQL_RENT, con=engine)
        print("  -> 'raw_trade' 로드 중...")
        df_trade = pd.read_sql(SQL_TRADE, con=engine)
        print("  -> 'raw_ledger' 로드 중...")
        df_ledger = pd.read_sql(SQL_LEDGER, con=engine)
    except Exception as e:
        print(f"DB 쿼리 중 치명적 오류 발생: {e}")
        raise

    print("--- 2. 데이터 정제 및 타입 변환 ---")

    # 2-1. 가격 변환
    df_rent['RENT_PRICE'] = pd.to_numeric(df_rent['RENT_PRICE'], errors='coerce')
    df_rent['MONTHLY_RENT'] = pd.to_numeric(df_rent['MONTHLY_RENT'].fillna(0), errors='coerce')
    df_trade['TRADE_PRICE'] = pd.to_numeric(df_trade['TRADE_PRICE'], errors='coerce')

    # 2-2. 날짜 변환
    df_rent['CONTRACT_DATE'] = pd.to_datetime(df_rent['CONTRACT_DATE'], errors='coerce')
    df_trade['TRADE_DATE'] = pd.to_datetime(df_trade['TRADE_DATE'], errors='coerce')
    df_ledger['USE_APR_DAY'] = pd.to_datetime(df_ledger['USE_APR_DAY'], errors='coerce')

    # 2-3. "전세" 계약만 필터링 (월세가 0인 계약)
    df_rent = df_rent[df_rent['MONTHLY_RENT'] == 0].copy()

    # 2-4. NULL 데이터 처리
    df_rent = df_rent.dropna(subset=['CONTRACT_DATE', 'RENT_PRICE'])
    df_trade = df_trade.dropna(subset=['TRADE_DATE', 'TRADE_PRICE'])

    print("--- 3. 고유 식별 키(key) 생성 중 ---")
    df_rent['key'] = df_rent.apply(_create_join_key, axis=1)
    df_trade['key'] = df_trade.apply(_create_join_key, axis=1)
    df_ledger['key'] = df_ledger.apply(_create_join_key, axis=1)

    # 중복 키 제거 (가장 최신 정보만 남김)
    df_trade = df_trade.sort_values(by='TRADE_DATE').drop_duplicates(subset=['key'], keep='last')
    df_ledger = df_ledger.drop_duplicates(subset=['key'], keep='last')
    df_rent = df_rent.dropna(subset=['key'])

    print("--- 4. 데이터 결합 (Merge) ---")

    # 4-1. 전세-매매 시계열 조인 (merge_asof)
    df_rent = df_rent.sort_values(by='CONTRACT_DATE')
    df_trade = df_trade.sort_values(by='TRADE_DATE')

    df_merged = pd.merge_asof(
        df_rent,
        df_trade[['key', 'TRADE_PRICE', 'TRADE_DATE']],
        left_on='CONTRACT_DATE',
        right_on='TRADE_DATE',
        by='key',
        direction='backward',
        tolerance=pd.Timedelta(days=365 * 2)  # (2년 내 매매가만 인정)
    )

    # 4-2. 건축물대장 결합 (key 기준)
    df_merged = pd.merge(df_merged, df_ledger, on='key', how='left')

    print("--- 5. 특성 공학 (Feature Engineering) ---")
    df_final = pd.DataFrame()

    # (1) 전세가율 (jeonse_ratio)
    df_final['jeonse_ratio'] = df_merged['RENT_PRICE'] / df_merged['TRADE_PRICE']

    # 0으로 나누는 경우(inf, -inf)를 처리하기 위해 replace 추가
    df_final['jeonse_ratio'] = df_final['jeonse_ratio'].replace([np.inf, -np.inf], 5.0)
    # JOIN 실패 시 NaN으로 유지
    df_final['jeonse_ratio'] = df_final['jeonse_ratio'].fillna(np.nan).clip(0, 1.5)

    # (2) 위반건축물 여부 (is_illegal_building)
    df_final['is_illegal_building'] = df_merged['IS_ILLEGAL'].apply(
        lambda x: 1 if str(x).upper() == 'Y' else 0
    )

    # (3) 건물 나이 (building_age)
    df_final['building_age'] = (df_merged['CONTRACT_DATE'] - df_merged['USE_APR_DAY']).dt.days / 365.25
    df_final['building_age'] = df_final['building_age'].fillna(0).clip(0, 100)

    # (4) 주용도 (building_use) - 원-핫 인코딩용
    df_merged['MAIN_PURPOSE'] = df_merged['MAIN_PURPOSE'].fillna('기타')

    categories = ['아파트', '다세대주택', '오피스텔', '근린생활시설', '기타']
    df_merged['MAIN_PURPOSE'] = df_merged['MAIN_PURPOSE'].replace('공동주택', '아파트')
    df_merged['MAIN_PURPOSE'] = df_merged['MAIN_PURPOSE'].apply(
        lambda x: '근린생활시설' if any(c in str(x) for c in ['근린', '판매', '교육연구', '종교']) else x
    )

    df_merged['building_use'] = pd.Categorical(
        df_merged['MAIN_PURPOSE'],
        categories=categories,
    ).fillna('기타')

    # prefix='building_use' 사용
    df_processed = pd.get_dummies(
        df_merged['building_use'],
        prefix='building_use',
        drop_first=False
    )
    df_final = pd.concat([df_final, df_processed], axis=1)


    # [가정 2] 선순위 대출은 매매가의 0~40% 사이에서 랜덤하게 발생
    random_loan_ratios = np.random.uniform(0, 0.4, size=len(df_merged))
    loan_amount = (df_merged['TRADE_PRICE'].fillna(0) * random_loan_ratios).fillna(0)

    # [가정 3] 부채+전세가율 (가장 중요한 특성)
    df_final['loan_plus_jeonse_ratio'] = \
        (loan_amount + df_merged['RENT_PRICE']) / df_merged['TRADE_PRICE']

    # inf, fillna, clip 적용
    df_final['loan_plus_jeonse_ratio'] = df_final['loan_plus_jeonse_ratio'].replace([np.inf, -np.inf], 5.0)
    # JOIN 실패 시 NaN으로 유지
    df_final['loan_plus_jeonse_ratio'] = df_final['loan_plus_jeonse_ratio'].fillna(np.nan)
    df_final['loan_plus_jeonse_ratio'] = df_final['loan_plus_jeonse_ratio'].clip(0, 1.5)

    return df_final