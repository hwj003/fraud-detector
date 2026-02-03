# fraud_detector_project/scripts/data_processor.py

import pandas as pd
import numpy as np  # inf 처리를 위해 numpy 임포트
import os
import sys

# --- [필수] 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# --- 중앙 설정 파일(engine) 임포트 ---
from app.core.database import engine
from app.services.feature_service import calculate_risk_features

# 4-1. 시세 추정 (기존 로직 유지하거나 함수화 가능, 여기선 DataFrame 연산 효율을 위해 유지)
def _estimate_market_price_row(row):
    if pd.notna(row['TRADE_PRICE']) and row['TRADE_PRICE'] > 0:
        return row['TRADE_PRICE']
    if pd.notna(row['PUBLIC_PRICE']) and row['PUBLIC_PRICE'] > 0:
        m_use = str(row['main_use'])
        if any(x in m_use for x in ['다세대', '오피스텔', '연립', '근린']):
            return row['PUBLIC_PRICE'] * 1.8
        return row['PUBLIC_PRICE'] * 1.5
    return np.nan

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

    df_merged['ESTIMATED_MARKET_PRICE'] = df_merged.apply(_estimate_market_price_row, axis=1)
    df_merged = df_merged.dropna(subset=['ESTIMATED_MARKET_PRICE'])

    # 4-2. [핵심 변경] calculate_risk_features 함수를 DataFrame 전체에 적용
    # DataFrame의 각 행(row)을 넘겨서 딕셔너리를 받은 뒤, 다시 DataFrame으로 변환

    def apply_feature_engineering(row):
        # 1. 단기 소유 가중치 계산 (날짜 차이)
        short_term_w = 0.0
        try:
            if pd.notna(row['ownership_changed_date']) and pd.notna(row['CONTRACT_DATE']):
                days = (row['CONTRACT_DATE'] - row['ownership_changed_date']).days
                if days < 90:
                    short_term_w = 0.3
                elif days < 730:
                    short_term_w = 0.1
        except:
            pass

        # 2. 신탁 여부
        is_trust = 1 if row['owner_name'] and '신탁' in str(row['owner_name']) else 0

        # 3. 위반 여부
        is_viol = 1 if str(row['is_violating_building']).strip() == 'Y' else 0

        # 4. 함수 호출
        feats = calculate_risk_features(
            deposit_amount=row['RENT_PRICE'],
            market_price=row['ESTIMATED_MARKET_PRICE'],
            real_debt=0,  # 학습 데이터엔 등기부 채권 정보가 보통 없음
            main_use=row['main_use'],
            usage_approval_date=row['use_apr_day'],  # datetime 객체도 처리되도록 함수 수정 필요할 수 있음
            is_illegal=is_viol,
            is_trust_owner=is_trust,
            short_term_weight=short_term_w,
            parking_count = row.get('parking_cnt', 0),  # df_title에서 가져온 값
            household_count = row.get('household_cnt', 0),  # df_title에서 가져온 값
        )
        return pd.Series(feats)

    # 새로운 피처들을 생성하여 기존 DF에 병합
    feature_df = df_merged.apply(apply_feature_engineering, axis=1)
    df_final = pd.concat([df_merged, feature_df], axis=1)

    print("--- [Data Augmentation] 가상의 빚(Debt) 데이터 주입 ---")
    # 1. 안전한 데이터 중 30%를 복제하여 '위험 데이터'로 변조
    safe_samples = df_final[df_final['total_risk_ratio'] < 0.7].sample(frac=0.3, random_state=42).copy()

    # 2. 가상의 빚을 시세의 50% ~ 90% 수준으로 랜덤하게 부여
    import numpy as np
    safe_samples['random_debt_ratio'] = np.random.uniform(0.5, 0.9, size=len(safe_samples))

    # 'real_debt' 피처를 역산해서 주입 (total_risk_ratio를 높이기 위해)
    # 주의: feature_engineering 로직상 total_risk_ratio는 (보증금 + 빚)/시세 임.
    # 여기서는 피처를 다시 계산해주는 것이 가장 정확함.

    def inject_fake_debt(row):
        # 가상의 빚 설정 (시세 * 랜덤비율)
        fake_debt = row['ESTIMATED_MARKET_PRICE'] * row['random_debt_ratio']

        # 피처 다시 계산
        feats = calculate_risk_features(
            deposit_amount=row['RENT_PRICE'],
            market_price=row['ESTIMATED_MARKET_PRICE'],
            real_debt=fake_debt,  # <--- 가짜 빚 주입!
            main_use=row['main_use'],
            usage_approval_date=row['use_apr_day'],
            is_illegal=int(row['is_violating_building'] == 'Y'),
            is_trust_owner=0,  # 일단 신탁은 아니라고 가정
            short_term_weight=0.0
        )
        # 정답지도 '위험'으로 강제 설정할 것이므로 덮어쓰기
        return pd.Series(feats)

    augmented_features = safe_samples.apply(inject_fake_debt, axis=1)
    augmented_df = pd.concat([safe_samples[df_merged.columns], augmented_features], axis=1)

    # 3. 원본 데이터와 합치기
    df_final = pd.concat([df_final, augmented_df], axis=0).reset_index(drop=True)

    print(f"-> 데이터 증강 완료: 총 {len(df_final)}건 (원본 + 가상채권데이터)")
    return df_final

if __name__ == "__main__":
    # 테스트 실행
    df = load_and_engineer_features()
    print(df.head())
    print("\n[Risk Level 분포]")
    print(df['total_risk_ratio'].describe())