# fraud_detector_project/scripts/data_processor.py

import pandas as pd
import numpy as np
import re
import os
import sys
from datetime import datetime

# --- 중앙 설정 파일(engine) 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
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
    [수정] DB의 raw 테이블 3개를 JOIN하여 모델 학습용 특성을 생성합니다.
    (등기부등본 'raw_registry'가 없으므로 불완전한 특성만 생성)
    """

    print("--- 1. 원본 데이터 로드 중 (DB) ---")

    # [주의] data_processor.py는 raw 테이블의 *컬럼명*을 정확히 알아야 합니다.
    # (fetch 스크립트에서 저장한 DB 컬럼명으로 수정)
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
    df_trade['TRADE_PRICE'] = pd.to_numeric(df_trade['TRADE_PRICE'], errors='coerce') * 10000  # (만원 -> 원 단위 통일)

    # 2-2. 날짜 변환
    df_rent['CONTRACT_DATE'] = pd.to_datetime(df_rent['CONTRACT_DATE'], errors='coerce')
    df_trade['TRADE_DATE'] = pd.to_datetime(df_trade['TRADE_DATE'], errors='coerce')
    df_ledger['USE_APR_DAY'] = pd.to_datetime(df_ledger['USE_APR_DAY'], errors='coerce')

    # 2-3. [중요] "전세" 계약만 필터링 (월세가 0인 계약)
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
        direction='backward',  # 전세 계약일보다 *과거*의 매매가를 찾음
        tolerance=pd.Timedelta(days=365 * 2)  # (최대 2년 내 매매가만 인정)
    )

    # 4-2. 건축물대장 결합 (key 기준)
    df_merged = pd.merge(df_merged, df_ledger, on='key', how='left')

    print("--- 5. 특성 공학 (Feature Engineering) ---")
    df_final = pd.DataFrame()

    # (1) 전세가율 (jeonse_ratio)
    # [경고] 선순위 대출이 없어 불완전함!
    df_final['jeonse_ratio'] = df_merged['RENT_PRICE'] / df_merged['TRADE_PRICE']
    # 매매가가 없거나(NaN), 0이거나, 전세가가 더 높으면(>1) -> 위험(1.0)으로 간주
    df_final['jeonse_ratio'] = df_final['jeonse_ratio'].fillna(1.0).clip(0, 5)
    df_final['jeonse_ratio'] = df_final['jeonse_ratio'].apply(lambda x: 1.0 if x > 1.0 else x)

    # (2) 위반건축물 여부 (is_illegal_building)
    df_final['is_illegal_building'] = df_merged['IS_ILLEGAL'].apply(
        lambda x: 1 if str(x).upper() == 'Y' else 0
    )

    # (3) 건물 나이 (building_age)
    df_final['building_age'] = (df_merged['CONTRACT_DATE'] - df_merged['USE_APR_DAY']).dt.days / 365.25
    df_final['building_age'] = df_final['building_age'].fillna(df_final['building_age'].mean()).clip(0, 100)

    # (4) 주용도 (building_use) - 원-핫 인코딩용
    df_merged['MAIN_PURPOSE'] = df_merged['MAIN_PURPOSE'].fillna('기타')

    categories = ['아파트', '다세대주택', '오피스텔', '근린생활시설', '기타']
    # '공동주택'은 '아파트'로 간주
    df_merged['MAIN_PURPOSE'] = df_merged['MAIN_PURPOSE'].replace('공동주택', '아파트')
    # '판매시설' 등은 '근린생활시설'로 간주
    df_merged['MAIN_PURPOSE'] = df_merged['MAIN_PURPOSE'].apply(
        lambda x: '근린생활시설' if any(c in str(x) for c in ['근린', '판매', '교육연구']) else x
    )

    df_merged['building_use'] = pd.Categorical(
        df_merged['MAIN_PURPOSE'],
        categories=categories,
        # categories에 없는 값은 '기타'로 처리
    ).fillna('기타')

    df_processed = pd.get_dummies(
        df_merged['building_use'],
        columns=['building_use'],
        drop_first=False
    )
    df_final = pd.concat([df_final, df_processed], axis=1)

    print(f"특성 공학 완료. 최종 생성된 특성 개수: {len(df_final.columns)}")
    print(df_final.head())

    # (경고) 등기부 데이터(신탁, 부채)가 누락됨
    df_final['has_trust'] = 0  # (데이터가 없으므로 0으로 채움)
    df_final['loan_plus_jeonse_ratio'] = df_final['jeonse_ratio']  # (부채가 없다고 가정)

    return df_final