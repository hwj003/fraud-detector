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
def _create_join_key_from_columns(row, keys=['시군구', '법정동', '본번', '부번']):
    """
    raw_rent, raw_trade 테이블용 키 생성 함수
    입력: 컬럼이 분리된 데이터
    출력: '인천광역시 부평구-부평동-0065-0124' 형태의 문자열
    """
    try:
        sgg = str(row[keys[0]]).strip()
        bjd = str(row[keys[1]]).strip()
        bon = str(row[keys[2]]).split('.')[0].zfill(4).strip()
        bu = str(row[keys[3]]).split('.')[0].zfill(4).strip()
        return f"{sgg}-{bjd}-{bon}-{bu}"
    except Exception as e:
        print(f"키 생성 중 알 수 없는 오류: {e}")
        return None

def _create_join_key_from_unique_no(unique_no):
    """
    building_info용 키 생성
    unique_number 포맷: '2823710100-3-00650124-101호' (SGG+BJD - GUBUN - BON+BU - HO)
    목표 포맷: '28237-10100-0065-0124' (raw_rent와 동일하게)
    """
    try:
        if pd.isna(unique_no) or unique_no == '':
            return None

        # 하이픈(-)으로 분리
        # parts[0]: '2823710100' (시군구5자리 + 법정동5자리)
        # parts[1]: '3'
        # parts[2]: '00650124' (본번4자리 + 부번4자리)
        parts = unique_no.split('-')

        if len(parts) < 3:
            return None

        part_sgg_bjd = parts[0] # 2823710100
        part_bon_bu = parts[2]  # 00650124

        if len(part_sgg_bjd) < 10 or len(part_bon_bu) < 8:
            return None

        sgg = part_sgg_bjd[:5]
        bjd = part_sgg_bjd[5:10]
        bon = part_bon_bu[:4]
        bu = part_bon_bu[4:8]

        return f"{sgg}-{bjd}-{bon}-{bu}"
    except Exception as e:
        return None


def _create_join_key_for_title(row):
    """
    [신규] building_title_info (표제부) 용 키 생성
    DB 컬럼: sigungu_code, bjdong_code, bunji ('402' or '402-1')
    목표: '28237-10100-0402-0000'
    """
    try:
        sgg = str(row['sigungu_code']).strip()
        bjd = str(row['bjdong_code']).strip()
        raw_bunji = str(row['bunji']).strip()

        if '-' in raw_bunji:
            bon, bu = raw_bunji.split('-')
        else:
            bon, bu = raw_bunji, '0'

        bon = bon.zfill(4)
        bu = bu.zfill(4)

        return f"{sgg}-{bjd}-{bon}-{bu}"
    except Exception:
        return None

def _create_join_key_from_address(address_str):
    """
    building_info 테이블용 키 생성 함수
    입력: '인천광역시 부평구 부평동 65-124' (문자열 지번주소)
    출력: '인천광역시 부평구-부평동-0065-0124' (위 함수와 동일한 포맷)
    """
    try:
        if pd.isna(address_str) or address_str == '':
            return None

        # 1. 마지막 공백을 기준으로 나눔 (앞부분: 시군구+동, 뒷부분: 번지)
        # 예: "인천광역시 부평구 부평동", "65-124"
        parts = address_str.rsplit(' ', 1)
        if len(parts) != 2:
            return None

        addr_part = parts[0] # 시군구 + 동
        bunji_part = parts[1] # 65-124

        # 2. 시군구와 동 분리 (마지막 공백 기준)
        # 예: "인천광역시 부평구", "부평동"
        addr_split = addr_part.rsplit(' ', 1)
        if len(addr_split) != 2:
            return None

        sgg = addr_split[0].strip()
        bjd = addr_split[1].strip()

        # 3. 본번, 부번 처리
        if '-' in bunji_part:
            bon, bu = bunji_part.split('-')
        else:
            bon, bu = bunji_part, '0'

        bon = bon.zfill(4)
        bu = bu.zfill(4)

        return f"{sgg}-{bjd}-{bon}-{bu}"
    except Exception:
        return None


def _extract_floor_from_detail(addr):
    """
    상세주소(예: '101동 302호', 'B01호')에서 층수를 추출하는 함수
    """
    try:
        # 숫자 추출
        import re
        if not isinstance(addr, str): return None

        val = 0

        # 전략 1: 명확하게 '호'가 붙은 숫자 찾기 (가장 정확)
        # 예: "101동 1501호" -> 1501 추출
        match_ho = re.search(r'(\d+)호', addr)
        if match_ho:
            val = int(match_ho.group(1))

        # 전략 2: '호'가 없다면, 주소의 가장 마지막 부분에서 숫자 찾기
        # 예: "110동 201" -> 201 추출
        else:
            # 공백으로 나눈 뒤 마지막 덩어리(Token) 선택
            tokens = addr.split()
            if tokens:
                last_token = tokens[-1]
                # 마지막 덩어리 안에서 숫자만 추출
                numbers = re.findall(r'\d+', last_token)
                if numbers:
                    val = int(numbers[-1])  # 여러 개면 그중 마지막 것
        # 층수 계산 로직 (공통)
        if val == 0: return 0
        if val < 100: return 1  # 1~99호는 1층으로 간주
        return val // 100  # 201 -> 2, 1501 -> 15

    except Exception as e:
        # print(f"층수 파싱 오류: {e}") # 디버깅 필요시 주석 해제
        return 0


# --- 2. 메인 데이터 가공 함수 ---
def load_and_engineer_features() -> pd.DataFrame:
    """
    DB의 raw 테이블(rent, trade)과 building_info, public_price_history를 JOIN하여
    전세사기 위험도 예측을 위한 모델 학습용 데이터를 생성합니다.
    """

    print("--- 1. 원본 데이터 로드 중 (DB) ---")

    # 1-1. 전세/월세 실거래가 (Target & Input)
    SQL_RENT = """
               SELECT 시군구, \
                      법정동, \
                      본번, \
                      부번, \
                      보증금 AS RENT_PRICE, \
                      월세  AS MONTHLY_RENT, \
                      계약일 AS CONTRACT_DATE, \
                      건물유형 AS BUILDING_TYPE, \
                      층 AS FLOOR, \
                      전용면적 AS AREA, \
                      건물명 AS BUILDING_NAME, \
                      건축년도 AS BUILDING_YEAR
               FROM raw_rent \
               WHERE 월세 = 0
                 AND 계약일 >= '20230101' -- 최근 2년 데이터만 사용
               """

    # 1-2. 매매 실거래가 (Market Price Reference)
    SQL_TRADE = """
                SELECT 시군구, \
                       법정동, \
                       본번, \
                       부번, \
                       거래금액 AS TRADE_PRICE, \
                       계약일  AS TRADE_DATE, \
                       전용면적 AS AREA, \
                       건축년도 AS BUILDING_AGE, \
                       건물유형 AS BUILDING_TYPE
                FROM raw_trade \
                WHERE 계약일 >= '20230101' -- 최근 2년 데이터만 사용
                """

    # 1-3. 건축물대장 정보
    SQL_BUILDING = """
                   SELECT id AS building_info_id,
                          unique_number,
                          lot_address,
                          main_use,
                          exclusive_area AS AREA,
                          owner_name,
                          ownership_changed_date,
                          detail_address,
                          is_violating_building
                   FROM building_info
                   WHERE unique_number IS NOT NULL
                   """

    # 1-4. 공시가격 히스토리
    SQL_PRICE_HISTORY = """
                        SELECT building_info_id,
                               price AS PUBLIC_PRICE,
                               base_date AS PRICE_DATE
                        FROM public_price_history
                        """

    # 1-5. 집합 건축물대장 표제부
    SQL_TITLE = """
                    SELECT 
                           unique_number,
                           sigungu_code, bjdong_code, bunji,
                           household_cnt,      -- 총 세대수
                           parking_cnt,        -- 주차대수
                           elevator_cnt,       -- 승강기대수
                           use_apr_day,        -- 사용승인일
                           grnd_flr_cnt,       -- 지상 층수
                           is_violating AS title_violation -- 표제부상 위반 여부
                    FROM building_title_info
                    """

    try:
        df_rent = pd.read_sql(SQL_RENT, con=engine)
        df_trade = pd.read_sql(SQL_TRADE, con=engine)
        df_building = pd.read_sql(SQL_BUILDING, con=engine)
        df_price = pd.read_sql(SQL_PRICE_HISTORY, con=engine)
        df_title = pd.read_sql(SQL_TITLE, con=engine)
    except Exception as e:
        print(f"DB 쿼리 중 치명적 오류 발생: {e}")
        raise

    print("--- 2. 데이터 정제 및 키 생성 ---")

    # 숫자형 변환
    df_rent['RENT_PRICE'] = pd.to_numeric(df_rent['RENT_PRICE'], errors='coerce')
    df_trade['TRADE_PRICE'] = pd.to_numeric(df_trade['TRADE_PRICE'], errors='coerce')
    df_rent['AREA'] = pd.to_numeric(df_rent['AREA'], errors='coerce')
    df_building['AREA'] = pd.to_numeric(df_building['AREA'], errors='coerce')
    df_price['PUBLIC_PRICE'] = pd.to_numeric(df_price['PUBLIC_PRICE'], errors='coerce') / 10000

    # 날짜형 변환
    df_rent['CONTRACT_DATE'] = pd.to_datetime(df_rent['CONTRACT_DATE'], errors='coerce')
    df_trade['TRADE_DATE'] = pd.to_datetime(df_trade['TRADE_DATE'], errors='coerce')
    df_price['PRICE_DATE'] = pd.to_datetime(df_price['PRICE_DATE'], errors='coerce')
    df_building['ownership_changed_date'] = pd.to_datetime(df_building['ownership_changed_date'], errors='coerce')

    # 키 생성 (개선된 함수 사용)
    col_map = {'sgg': '시군구', 'bjd': '법정동', 'bon': '본번', 'bu': '부번'}
    df_rent['key'] = df_rent.apply(lambda row: _create_join_key_from_columns(row), axis=1)
    df_trade['key'] = df_trade.apply(lambda row: _create_join_key_from_columns(row), axis=1)
    df_building['key'] = df_building['unique_number'].apply(_create_join_key_from_unique_no)

    # 유효한 키만 남기기
    df_rent = df_rent.dropna(subset=['key', 'RENT_PRICE'])
    df_building = df_building.dropna(subset=['key'])

    # 표제부(사용승인일)를 building_info에 미리 결합 (19자리 PNU 기준)
    # df_title의 unique_number가 19자리라고 가정
    df_building['pnu_19'] = df_building['unique_number'].astype(str).str.slice(0, 19)
    df_title['pnu_19'] = df_title['unique_number'].astype(str).str.slice(0, 19)

    # 중복 제거 (건물당 1개)
    df_title = df_title.drop_duplicates(subset=['pnu_19'])
    df_building = pd.merge(df_building, df_title[['pnu_19', 'use_apr_day']], on='pnu_19', how='left')

    print("--- 3. 데이터 결합 및 필터링 (Merge & Filter) ---")

    # (1) 전세 + 매매 (merge_asof: 날짜 근접 매칭)
    df_rent = df_rent.sort_values('CONTRACT_DATE')
    df_trade = df_trade.sort_values('TRADE_DATE')

    df_merged = pd.merge_asof(
        df_rent,
        df_trade[['key', 'TRADE_PRICE', 'TRADE_DATE']],
        left_on='CONTRACT_DATE',
        right_on='TRADE_DATE',
        by='key',
        direction='backward',
        tolerance=pd.Timedelta(days=365 * 2)  # 2년 내 매매가 참조
    )

    # (2) 전세 + 건축물대장 (Left Join)
    # 학습용이므로, 건물이 매칭된 데이터만 살립니다 (Inner Join과 유사 효과를 위해 dropna)
    df_merged = pd.merge(df_merged, df_building, on='key', how='left', suffixes=('', '_BUILD'))

    # 건축물대장 정보가 없는 데이터 삭제 (분석 불가)
    df_merged = df_merged.dropna(subset=['building_info_id'])

    # [핵심] 면적 오차 필터링 (Area Mismatch Removal)
    # 아파트 전세가(30평)가 빌라 건물(10평)에 붙는 오류 제거
    # 오차 범위: 3.3m² (1평) 미만인 것만 유효
    df_merged['area_diff'] = abs(df_merged['AREA'] - df_merged['AREA_BUILD'])

    initial_count = len(df_merged)
    df_merged = df_merged[df_merged['area_diff'] < 3.3].copy()
    print(f"-> 면적 불일치 데이터 {initial_count - len(df_merged)}건 제거됨")

    df_merged['building_info_id'] = df_merged['building_info_id'].astype(int)
    df_price['building_info_id'] = df_price['building_info_id'].astype(int)

    # (3) 공시지가 결합
    df_price = df_price.sort_values('PRICE_DATE')
    df_merged = pd.merge_asof(
        df_merged.sort_values('CONTRACT_DATE'),
        df_price,
        left_on='CONTRACT_DATE',
        right_on='PRICE_DATE',
        by='building_info_id',
        direction='backward'
    )

    print("--- 4. 파생변수 생성 및 시뮬레이션 (Modified) ---")

    # 4-1. 시세 추정
    def estimate_market_price(row):
        if pd.notna(row['TRADE_PRICE']) and row['TRADE_PRICE'] > 0:
            return row['TRADE_PRICE']
        if pd.notna(row['PUBLIC_PRICE']) and row['PUBLIC_PRICE'] > 0:
            m_use = str(row['main_use'])
            if any(x in m_use for x in ['다세대', '오피스텔', '연립', '근린']):
                return row['PUBLIC_PRICE'] * 1.8
            return row['PUBLIC_PRICE'] * 1.5
        return np.nan

    df_merged['ESTIMATED_MARKET_PRICE'] = df_merged.apply(estimate_market_price, axis=1)
    df_merged = df_merged.dropna(subset=['ESTIMATED_MARKET_PRICE'])

    # 4-2. 전세가율
    df_merged['jeonse_ratio'] = df_merged['RENT_PRICE'] / df_merged['ESTIMATED_MARKET_PRICE']
    valid_mask = df_merged['jeonse_ratio'] < 2.0
    df_final = df_merged[valid_mask].copy()

    # 4-3. HUG 위험도
    filled_public = df_final['PUBLIC_PRICE'].fillna(df_final['ESTIMATED_MARKET_PRICE'] * 0.7)
    hug_limit = filled_public * 1.26
    df_final['hug_risk_ratio'] = df_final.apply(
        lambda x: x['RENT_PRICE'] / hug_limit[x.name] if hug_limit[x.name] > 0 else 0, axis=1
    )

    # 4-4. [수정됨] 위험 요소 점수화 (Feature Engineering)
    # 이 값들을 더 이상 '대출금'으로 환산해서 빚에 더하지 않습니다.
    # 대신 AI가 학습할 '특징(Feature)'으로만 남겨둡니다.

    def type_weight(use):
        s = str(use)
        if '근린' in s: return 0.4
        if any(c in s for c in ['다세대', '오피스텔', '연립']): return 0.1
        return 0.0

    def trust_weight(owner):
        return 0.5 if owner and ('신탁' in str(owner)) else 0.0

    def calc_short_term_weight(row):
        try:
            if pd.isna(row['ownership_changed_date']): return 0.0
            # use_apr_day나 contract_date 등 기준 날짜 설정 필요 (여기선 현재 시점 혹은 계약일 기준)
            # 학습 데이터엔 'CONTRACT_DATE'가 있으므로 사용
            contract_date = row.get('CONTRACT_DATE', datetime.now())

            days = (contract_date - row['ownership_changed_date']).days

            if days < 90: return 0.3
            if days < 730: return 0.1
            return 0.0
        except:
            return 0.0

    w_type = df_final['main_use'].apply(type_weight)
    w_trust = df_final['owner_name'].apply(trust_weight)

    df_final['is_trust_owner'] = df_final['owner_name'].apply(lambda x: 1 if '신탁' in str(x) else 0)
    df_final['short_term_weight'] = df_final.apply(calc_short_term_weight, axis=1)

    # 랜덤값은 제거하거나, 노이즈 추가용으로만 미세하게 사용
    # 여기서는 AI 학습용이므로 랜덤값은 제거하고 정직한 스코어만 남깁니다.
    df_final['risk_factor_score'] = (w_type + w_trust).clip(0, 1.0)

    # [중요] 기존 코드와의 호환성을 위해 'estimated_loan_ratio'라는 컬럼명은 유지하되,
    # 그 의미를 '위험 점수'로 변경합니다.
    df_final['estimated_loan_ratio'] = df_final['risk_factor_score']

    # 4-5. [수정됨] 깡통전세 위험도 (Label/Target 관련)
    # 실제 빚(Real Debt)이 없으므로 전세가율과 동일하게 설정합니다.
    # AI는 "risk_factor_score가 높고 & jeonse_ratio가 높을 때" 위험하다는 패턴을 스스로 찾게 됩니다.
    df_final['total_risk_ratio'] = df_final['RENT_PRICE'] / df_final['ESTIMATED_MARKET_PRICE']

    # 4-6. 기타 파생변수
    def simplify_use(x):
        s = str(x)
        if '아파트' in s: return 'APT'
        if '오피스텔' in s: return 'OFFICETEL'
        if '다세대' in s or '연립' in s: return 'VILLA'
        return 'ETC'

    df_final['simple_type'] = df_final['main_use'].apply(simplify_use)
    df_final['is_illegal'] = df_final['is_violating_building'].apply(lambda x: 1 if str(x).strip() == 'Y' else 0)

    # building_age 등 나머지 로직 유지
    df_final['use_apr_day'] = pd.to_datetime(df_final['use_apr_day'], errors='coerce')
    df_final['building_age'] = (datetime.now() - df_final['use_apr_day']).dt.days / 365.25
    df_final['building_age'] = df_final['building_age'].fillna(0)

    # 5. 최종 컬럼 정리
    final_cols = [
        'RENT_PRICE', 'ESTIMATED_MARKET_PRICE', 'PUBLIC_PRICE',
        'jeonse_ratio', 'hug_risk_ratio',
        'total_risk_ratio',  # 순수 (전세+실제빚)/시세
        'estimated_loan_ratio',  # (=risk_factor_score) 정성적 위험 점수
        'building_age', 'is_illegal', 'simple_type', 'main_use',
        'is_trust_owner', 'short_term_weight'
    ]

    df_result = df_final[final_cols].copy()
    print(f"--- 데이터 가공 완료: {len(df_result)}건 생성 ---")
    return df_result

if __name__ == "__main__":
    # 테스트 실행
    df = load_and_engineer_features()
    print(df.head())
    print("\n[Risk Level 분포]")
    print(df['total_risk_ratio'].describe())