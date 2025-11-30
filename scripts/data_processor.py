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
                      층 AS FLOOR
               FROM raw_rent \
               """

    # 1-2. 매매 실거래가 (Market Price Reference)
    SQL_TRADE = """
                SELECT 시군구, \
                       법정동, \
                       본번, \
                       부번, \
                       `거래금액(만원)` AS TRADE_PRICE, \
                       계약일        AS TRADE_DATE
                FROM raw_trade \
                """

    # 1-3. 건축물대장 정보
    SQL_BUILDING = """
                   SELECT id AS building_info_id,
                          unique_number,
                          lot_address,
                          main_use,
                          exclusive_area,
                          owner_name,
                          ownership_changed_date,
                          detail_address,
                          is_violating_building
                   FROM building_info
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
                    SELECT sigungu_code, bjdong_code, bunji,
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

    print("--- 2. 데이터 정제 및 타입 변환 ---")

    # 2-1. 가격 변환
    df_rent['RENT_PRICE'] = pd.to_numeric(df_rent['RENT_PRICE'], errors='coerce')
    df_rent['MONTHLY_RENT'] = pd.to_numeric(df_rent['MONTHLY_RENT'].fillna(0), errors='coerce')
    df_trade['TRADE_PRICE'] = pd.to_numeric(df_trade['TRADE_PRICE'], errors='coerce')
    df_price['PUBLIC_PRICE'] = pd.to_numeric(df_price['PUBLIC_PRICE'], errors='coerce') / 10000

    # 층수(FLOOR) 정수형 변환 (소수점 제거)
    df_rent['FLOOR'] = pd.to_numeric(df_rent['FLOOR'], errors='coerce').fillna(0).astype(int)

    # 2-2. 날짜 변환
    df_rent['CONTRACT_DATE'] = pd.to_datetime(df_rent['CONTRACT_DATE'], errors='coerce')
    df_trade['TRADE_DATE'] = pd.to_datetime(df_trade['TRADE_DATE'], errors='coerce')
    df_price['PRICE_DATE'] = pd.to_datetime(df_price['PRICE_DATE'], errors='coerce')
    df_title['use_apr_day'] = pd.to_datetime(df_title['use_apr_day'], errors='coerce')
    df_building['ownership_changed_date'] = pd.to_datetime(df_building['ownership_changed_date'], errors='coerce')

    # 전세계약 필터링
    df_rent = df_rent[df_rent['MONTHLY_RENT'] == 0].copy()
    df_rent = df_rent.dropna(subset=['CONTRACT_DATE', 'RENT_PRICE'])
    df_price = df_price.sort_values('PRICE_DATE')

    # 키 생성
    df_rent['key'] = df_rent.apply(_create_join_key_from_columns, axis=1)
    df_trade['key'] = df_trade.apply(_create_join_key_from_columns, axis=1)
    df_title['key'] = df_title.apply(_create_join_key_for_title, axis=1)
    df_building['key'] = df_building['unique_number'].apply(_create_join_key_from_unique_no)
    # 3-3. 키 없는 데이터 제거 및 중복 제거
    df_rent = df_rent.dropna(subset=['key'])
    df_trade = df_trade.dropna(subset=['key']).sort_values('TRADE_DATE')

    # 중복 키 제거 (가장 최신 정보만 남김)
    df_rent = df_rent.dropna(subset=['key'])
    df_trade = df_trade.dropna(subset=['key']).sort_values(by='TRADE_DATE')
    df_building = df_building.dropna(subset=['key']).drop_duplicates(subset=['key'], keep='last')
    # 표제부는 동별 데이터이므로 키 중복 제거 (가장 최신 정보 유지)
    df_title = df_title.dropna(subset=['key']).drop_duplicates(subset=['key'], keep='last')

    # 건축물대장 데이터에서 층수 추출
    df_building['FLOOR'] = df_building['detail_address'].apply(_extract_floor_from_detail)

    # 층수 추출 실패한 데이터(None)는 삭제 처리
    df_building = df_building.dropna(subset=['FLOOR'])
    df_building['FLOOR'] = df_building['FLOOR'].astype(int)

    print("--- 4. 데이터 결합 (Merge) ---")

    # 4-1. 전세(Rent) + 매매(Trade) 결합 (시세 비교용)
    df_merged = pd.merge_asof(
        df_rent.sort_values('CONTRACT_DATE'),
        df_trade[['key', 'TRADE_PRICE', 'TRADE_DATE']],
        left_on='CONTRACT_DATE',
        right_on='TRADE_DATE',
        by='key',
        direction='backward',
        tolerance=pd.Timedelta(days=365 * 2)  # 2년 내 매매가 참조
    )

    # 4-2. 전세 + 건축물대장 결합
    df_merged = pd.merge(df_merged, df_building, on=['key', 'FLOOR'], how='inner')

    # 4-2-B. 전세 + 표제부(Title Info) 결합
    # 이제 나이, 세대수, 주차장 정보는 여기서 옵니다.
    df_merged = pd.merge(df_merged, df_title.drop(columns=['sigungu_code', 'bjdong_code', 'bunji']), on='key', how='left')

    # 4-3. 전세 + 공시가격
    df_merged_with_id = df_merged.dropna(subset=['building_info_id']).sort_values('CONTRACT_DATE')
    df_merged_with_id['building_info_id'] = df_merged_with_id['building_info_id'].astype(int)

    # 공시가격 붙이기 (계약일 기준 가장 최근 공시가격)
    df_final_temp = pd.merge_asof(
        df_merged_with_id,
        df_price[['building_info_id', 'PUBLIC_PRICE', 'PRICE_DATE']],
        left_on='CONTRACT_DATE',
        right_on='PRICE_DATE',
        by='building_info_id',
        direction='backward'
    )

    df_merged = df_final_temp

    # 시각화에 필요한 원본 칼럼들을 미리 복사해서 df_final 생성
    required_cols = [
        'RENT_PRICE', 'TRADE_PRICE', 'PUBLIC_PRICE', 'ESTIMATED_MARKET_PRICE',
        'CONTRACT_DATE', 'main_use', 'building_info_id',
        'household_cnt', 'parking_cnt', 'elevator_cnt', 'use_apr_day',
        'owner_name', 'ownership_changed_date'
    ]
    # 존재하는 칼럼만 복사 (오류 방지)
    existing_cols = [col for col in required_cols if col in df_merged.columns]
    df_final = df_merged[existing_cols].copy()

    # (1) 전세가율 (jeonse_ratio)
    # 매매가가 없으면 공시가격 * 1.4배를 추정 시세로 사용
    df_merged['ESTIMATED_MARKET_PRICE'] = df_merged['TRADE_PRICE'].fillna(df_merged['PUBLIC_PRICE'] * 1.4)
    df_final['ESTIMATED_MARKET_PRICE'] = df_merged['ESTIMATED_MARKET_PRICE']

    df_final['jeonse_ratio'] = df_merged['RENT_PRICE'] / df_merged['ESTIMATED_MARKET_PRICE']
    # 0으로 나누는 경우(inf, -inf)를 처리하기 위해 replace 추가
    df_final['jeonse_ratio'] = df_final['jeonse_ratio'].replace([np.inf, -np.inf], 5.0).fillna(0).clip(0, 1.5)

    # (2) HUG 보증보험 기준 위험도 (Rent  / (Public Price * 126%))
    # 가입한도 = (공시가격 * 140%) * 90%
    # 이 값이 1.0을 넘으면 보증보험 가입 불가 = 고위험
    hug_limit = df_merged['PUBLIC_PRICE'] * 1.26
    df_final['hug_risk_ratio'] = df_merged['RENT_PRICE'] / hug_limit
    df_final['hug_risk_ratio'] = df_final['hug_risk_ratio'].replace([np.inf, -np.inf], 5.0).fillna(0)

    # (3) 위반건축물 여부 (전유부 위반 OR 표제부 위반)
    # 하나라도 'Y'면 위반으로 간주
    is_vio_exclusive = df_merged['is_violating_building'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)
    is_vio_title = df_merged['title_violation'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)
    df_final['is_illegal_building'] = ((is_vio_exclusive == 1) | (is_vio_title == 1)).astype(int)

    # (4) 건물 나이 (표제부 use_apr_day 기준)
    df_final['building_age'] = (df_merged['CONTRACT_DATE'] - df_merged['use_apr_day']).dt.days / 365.25
    df_final['building_age'] = df_final['building_age'].fillna(0).clip(0, 50)  # 데이터 없으면 0 처리

    # (5) 세대당 주차대수 (Parking per Household)
    # 주차난이 심할수록(0.5대 미만) 가격 방어가 안 되므로 리스크 요인
    df_final['parking_per_household'] = df_merged['parking_cnt'] / df_merged['household_cnt']
    df_final['parking_per_household'] = df_final['parking_per_household'].replace([np.inf, -np.inf], 0).fillna(0)

    # (6) 나홀로 아파트 여부 (Scale Risk)
    # 100세대 미만이면 1, 아니면 0 (환금성 리스크)
    df_final['is_micro_complex'] = df_merged['household_cnt'].apply(lambda x: 1 if x < 100 else 0)

    # (7) 주용도 (building_use) - 원-핫 인코딩용
    df_merged['main_use'] = df_merged['main_use'].fillna('기타')

    # 2. 명칭 통일 로직 (포함 여부로 판단)
    def standardize_building_type(x):
        s = str(x)
        if '아파트' in s: return '아파트'  # 공동주택(아파트) -> 아파트
        if '오피스텔' in s: return '오피스텔'  # 업무시설(오피스텔) -> 오피스텔
        if '다세대' in s: return '다세대주택'
        if '연립' in s: return '연립주택'
        # 근린생활시설 관련 키워드 통합
        if any(c in s for c in ['근린', '소매', '사무', '점포']): return '근린생활시설'
        return '기타'

    df_merged['main_use'] = df_merged['main_use'].apply(standardize_building_type)

    df_final['main_use'] = df_merged['main_use']

    categories = ['아파트', '다세대주택', '오피스텔', '근린생활시설', '기타']
    df_merged['cat_use'] = pd.Categorical(df_merged['main_use'], categories=categories).fillna('기타')

    df_processed = pd.get_dummies(df_merged['cat_use'], prefix='use', drop_first=False)
    df_final = pd.concat([df_final, df_processed], axis=1)

    # [가상 선순위 대출 시뮬레이션]
    # 근거: 국토교통부 전세사기 피해 실태조사 보고서 (2025.06)

    # 0. 기본 불확실성
    # 모든 집에 대출이 있을 수도, 없을 수도 있다는 현실적 불확실성 반영 (0~20%)
    base_loan_ratio = np.random.uniform(0, 0.2, size=len(df_merged))

    # Logic 1: 건물 유형 가중치 (Type Weight)
    # 근거: 피해자의 86%가 비아파트(다세대, 오피스텔)에 집중됨. 아파트는 안전.
    type_weight = df_merged['main_use'].apply(
        lambda x: 0.2 if any(c in str(x) for c in ['다세대', '오피스텔', '근린', '연립']) else 0.0
    )

    # Logic 2: 가격 역상관성 가중치 (Price Inverse Weight)
    # 근거: 피해 보증금의 97%가 3억 원 이하 저가 주택. (무자본 갭투자 타겟)
    # 하위 30% 가격대 매물에 대해 대출 위험도 추가
    price_threshold_low = df_merged['ESTIMATED_MARKET_PRICE'].quantile(0.3)

    price_weight = df_merged['ESTIMATED_MARKET_PRICE'].apply(
        lambda x: 0.15 if x <= price_threshold_low else 0.0
    )

    # Logic 4: 위험 지역 가중치 (Regional Risk Weight)
    # 근거: 수원, 미추홀, 강서, 관악, 대전 등 특정 지역에 대규모 피해 집중
    # 위험 지역 키워드 (보고서 기반)
    high_risk_regions = ['미추홀', '강서', '관악', '수원', '대전', '대구']

    # Logic 5: 신탁 등기 위험
    # 근거: 신탁회사가 소유주일 경우, 일반 계약은 무효일 확률 높음
    df_final['is_trust_owner'] = df_merged['owner_name'].apply(
        lambda x: 1 if x and ('신탁' in str(x) or '자산관리' in str(x)) else 0
    )
    # 신탁이면 대출 비율을 확 높여서 위험군으로 분류되게 함 (+30%)
    trust_weight = df_final['is_trust_owner'] * 0.3

    # Logic 6: 단기 소유 위험
    # 근거: 계약일 기준 2년 이내에 주인이 바뀌었다면 갭투자/바지사장 의심
    def calc_ownership_duration(row):
        try:
            # 계약일 - 소유권변동일
            contract = row['CONTRACT_DATE']
            changed = row['ownership_changed_date']
            if pd.isna(contract) or pd.isna(changed): return 9999  # 알 수 없음

            days = (contract - changed).days
            return days
        except:
            return 9999

    df_merged['CONTRACT_DATE'] = pd.to_datetime(df_merged['CONTRACT_DATE'], errors='coerce')
    df_building['ownership_changed_date'] = pd.to_datetime(df_building['ownership_changed_date'], errors='coerce')

    df_final['ownership_days'] = df_merged.apply(calc_ownership_duration, axis=1)

    # 가중치 로직 (계약일 - 소유권 변동일)
    # 값이 음수인 경우: 세입자가 계약을 하고 입주한 뒤에 집주인이 바뀌었다는 의미
    # 값이 양수인 경우: 집 주인이 먼저 집을 샀고, 그 이후에 세입자와 전세 계약을 맺은 상태
    def calculate_short_term_weight(days):
        # 데이터 없음 (안전 간주)
        if days == 9999:
            return 0.0

        # 1. 동시진행 의심 구간 (계약일 전후 3개월)
        # 예: 계약일(1월 1일) ~ 변동일(4월 1일) 사이 -> -90 < days < 90
        if -90 < days < 90:
            return 0.3  # [최고 위험] 동시진행 강력 의심

        # 2. 계약 후 주인이 바뀐 경우 (음수)
        # 이미 살고 있는데 주인이 바뀜 -> 갭투자 매물일 가능성 높음
        elif days <= -90:
            return 0.15  # [위험] 갭투자 승계 (새 집주인 리스크)

        # 3. 계약 전 단기 매수 (양수 2년 미만)
        # 집을 산 지 얼마 안 돼서 전세 놓음 -> 갭투자 의심
        elif 0 <= days < 730:
            return 0.15  # [주의] 단기 보유

        # 4. 장기 보유 (안전)
        else:
            return 0.0

    short_term_weight = df_final['ownership_days'].apply(calculate_short_term_weight)

    df_final['short_term_weight'] = short_term_weight

    # 시군구 + 법정동 문자열 결합
    full_address = df_merged['시군구'].fillna('') + ' ' + df_merged['법정동'].fillna('')

    region_weight = full_address.apply(
        lambda x: 0.15 if any(region in str(x) for region in high_risk_regions) else 0.0
    )

    # --------------------------------------------------------------------------
    # 최종 대출 비율 산정 및 적용
    # --------------------------------------------------------------------------
    df_final['estimated_loan_ratio'] = (
            base_loan_ratio + type_weight + price_weight + region_weight + trust_weight + short_term_weight
    ).clip(0, 0.9)  # 최대 90% 제한

    # (5) 깡통전세 위험도 ( (대출 + 전세) / 추정시세 )
    loan_amount = df_merged['ESTIMATED_MARKET_PRICE'] * df_final['estimated_loan_ratio']
    df_final['total_risk_ratio'] = (loan_amount + df_merged['RENT_PRICE']) / df_merged['ESTIMATED_MARKET_PRICE']
    df_final['total_risk_ratio'] = df_final['total_risk_ratio'].clip(0, 2.0)

    # 이상치 처리
    df_final['total_risk_ratio'] = df_final['total_risk_ratio'].replace([np.inf, -np.inf], 0).fillna(0).clip(0, 2.0)

    print(f"--- 데이터 가공 완료: {len(df_final)}건 생성 ---")
    return df_final

if __name__ == "__main__":
    load_and_engineer_features()