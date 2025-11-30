import pandas as pd
import numpy as np
import joblib
import os
import sys
import re
from datetime import datetime

# --- 프로젝트 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- 모듈 임포트 ---
from scripts.fetch_ledger_exclusive import fetch_final_data_step, parse_and_save, get_dong_list_step, get_ho_list_step
from scripts.fetch_ledger_title import fetch_step1_search, fetch_step2_detail, parse_and_save_title
from scripts.kakao_localmap_api import get_road_address_from_kakao, get_building_name_from_kakao
from scripts.data_processor import (
    _create_join_key_from_unique_no, _extract_floor_from_detail, engine
)

# --- 모델 및 설정 로드 ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'fraud_rf_model.pkl')
try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"모델 로드 성공: {MODEL_PATH}")
except:
    print("모델 파일이 없습니다. 먼저 학습(train_model.py)을 실행하세요.")
    sys.exit(1)

# 학습 때 사용한 컬럼 순서 (매우 중요! 순서 틀리면 예측 엉망됨)
# train_model.py에서 학습할 때 썼던 features 리스트와 똑같아야 함
MODEL_FEATURES = [
    'jeonse_ratio', 'hug_risk_ratio', 'total_risk_ratio', 'building_age',
    'parking_per_household', 'is_micro_complex', 'estimated_loan_ratio',
    'is_trust_owner', 'short_term_weight'
]
# One-Hot Encoding용 기본 컬럼들
USE_COLS = ['use_아파트', 'use_오피스텔', 'use_다세대주택', 'use_근린생활시설', 'use_기타']


def normalize_address(address):
    """
    주소 문자열의 앞부분(시/도)을 정식 명칭으로 변환하는 함수
    예: "인천 부평구..." -> "인천광역시 부평구..."
    """
    # 1. 줄임말 매핑 테이블 (필요한 만큼 추가 가능)
    sido_map = {
        "서울": "서울특별시","서울시": "서울특별시","인천": "인천광역시","인천시": "인천광역시","경기": "경기도",
        "부산": "부산광역시","대구": "대구광역시","광주": "광주광역시","대전": "대전광역시","울산": "울산광역시",
        "세종": "세종특별자치시","강원": "강원특별자치도","충북": "충청북도","충남": "충청남도","전북": "전북특별자치도",
        "전남": "전라남도","경북": "경상북도","경남": "경상남도","제주": "제주특별자치도"
    }

    # 2. 주소가 비어있으면 그대로 반환
    if not address or not isinstance(address, str):
        return address

    # 3. 공백 기준으로 단어 분리
    tokens = address.split()

    if not tokens:
        return address

    # 4. 첫 번째 단어(시/도)가 매핑 테이블에 있는지 확인하고 교체
    first_word = tokens[0]

    # "인천" -> "인천광역시"
    if first_word in sido_map:
        tokens[0] = sido_map[first_word]

    # 5. 다시 합쳐서 반환
    return " ".join(tokens)

def get_real_time_data(address, deposit_amount):
    """
        주소를 받아서 DB 조회 또는 API 실시간 수집 후
        모델 입력용 데이터(Dictionary)로 변환하여 반환
        """
    print(f"\n🔎 분석 요청: {address} (보증금: {deposit_amount:,}원)")

    # 1. 주소 변환
    road_addr = get_road_address_from_kakao(address)
    if not road_addr:
        return None, "주소를 찾을 수 없습니다."
    # 주소 변환: 인천 => 인천광역시, 서울 => 서울특별시 등
    road_addr=normalize_address(road_addr)

    # 2. DB 조회 (이미 수집된 데이터인지 확인)
    # 여기서는 편의상 DB 쿼리로 가져오는 로직을 구현 (없으면 API 수집 로직 연결 필요)
    # 실제 서비스에선 API 수집 로직을 여기에 통합해야 함

    query = f"""
        SELECT 
            b.unique_number, b.detail_address, b.main_use, b.exclusive_area, 
            b.owner_name, b.ownership_changed_date, b.is_violating_building,
            p.price as PUBLIC_PRICE,
            t.household_cnt, t.parking_cnt, t.elevator_cnt, t.use_apr_day, t.is_violating as title_violation
        FROM building_info b
        LEFT JOIN public_price_history p ON b.id = p.building_info_id
        LEFT JOIN building_title_info t ON b.unique_number LIKE substr(t.unique_number, 1, 14) || '%' 
        WHERE b.road_address LIKE '%{road_addr}%' OR b.lot_address LIKE '%{address}%'
        ORDER BY p.base_date DESC LIMIT 1
    """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        return None, f"DB 조회 오류: {e}"

    if df.empty:
        return None, "데이터가 없습니다. (수집 필요)"
        # TODO: 여기서 fetch_ledger_... 함수들을 호출해서 실시간 수집 수행 가능
        # 3. 변수 가공 (Data Processor 로직 재사용)

    row = df.iloc[0]

    # (1) 기본 정보
    public_price = float(row['PUBLIC_PRICE']) / 10000  # 만원 단위
    deposit = deposit_amount / 10000  # 만원 단위

    # (2) 추정 시세 (공시가 * 1.4)
    estimated_market_price = public_price * 1.4

    # (3) 비율 계산
    jeonse_ratio = deposit / estimated_market_price
    hug_limit = public_price * 1.26
    hug_risk_ratio = deposit / hug_limit

    # (4) 건물 정보
    # 나이
    use_apr_day = pd.to_datetime(row['use_apr_day'])
    building_age = (datetime.now() - use_apr_day).days / 365.25 if pd.notnull(use_apr_day) else 10

    # 주차
    house_cnt = row['household_cnt'] if row['household_cnt'] > 0 else 1
    parking_per_household = row['parking_cnt'] / house_cnt

    # 나홀로 아파트
    is_micro = 1 if house_cnt < 100 else 0

    # (5) 리스크 가중치 (간소화된 로직 적용)
    # 신탁
    is_trust = 1 if '신탁' in str(row['owner_name']) else 0

    # 단기 소유
    try:
        own_date = pd.to_datetime(row['ownership_changed_date'])
        own_days = (datetime.now() - own_date).days
    except:
        own_days = 9999

    short_term_w = 0.0
    if own_days < 90:
        short_term_w = 0.3
    elif own_days < 730:
        short_term_w = 0.15

    # 추정 대출 비율 (간략화)
    base_loan = 0.2
    type_w = 0.2 if '아파트' not in str(row['main_use']) else 0.0
    est_loan_ratio = min(0.9, base_loan + type_w + short_term_w + (0.3 if is_trust else 0))

    # 깡통전세 비율
    loan_amt = estimated_market_price * est_loan_ratio
    total_risk_ratio = (loan_amt + deposit) / estimated_market_price

    # (6) One-Hot Encoding 준비
    main_use = str(row['main_use'])
    use_dict = {col: 0 for col in USE_COLS}

    if '아파트' in main_use:
        use_dict['use_아파트'] = 1
    elif '오피스텔' in main_use:
        use_dict['use_오피스텔'] = 1
    elif '다세대' in main_use:
        use_dict['use_다세대주택'] = 1
    elif any(c in main_use for c in ['근린', '소매']):
        use_dict['use_근린생활시설'] = 1
    else:
        use_dict['use_기타'] = 1

    # 4. 최종 입력 데이터 생성
    input_data = {
        'jeonse_ratio': jeonse_ratio,
        'hug_risk_ratio': hug_risk_ratio,
        'total_risk_ratio': total_risk_ratio,
        'building_age': building_age,
        'parking_per_household': parking_per_household,
        'is_micro_complex': is_micro,
        'estimated_loan_ratio': est_loan_ratio,
        'is_trust_owner': is_trust,
        'short_term_weight': short_term_w
    }
    input_data.update(use_dict)  # 원핫 컬럼 병합

    return input_data, row['detail_address']


def predict_risk(address, deposit_amount):
    # 1. 데이터 준비
    input_dict, bldg_name = get_real_time_data(address, deposit_amount)

    if not input_dict:
        return {"error": bldg_name}  # 에러 메시지 반환

    # 2. DataFrame 변환 (모델 입력용)
    # 학습 때와 컬럼 순서를 완벽하게 맞춰야 함
    df_input = pd.DataFrame([input_dict])

    # [핵심 수정] 모델이 학습할 때 썼던 피처 순서를 그대로 가져와서 정렬합니다.
    # (모델 파일 안에 저장되어 있는 정답 순서입니다)
    try:
        # 학습된 피처 이름 목록 가져오기
        train_features = rf_model.feature_names_in_

        # 해당 순서대로 데이터프레임 재배열 (없는 컬럼은 0으로 채움)
        df_input = df_input.reindex(columns=train_features, fill_value=0)

    except AttributeError:
        # 구버전 scikit-learn이거나 피처 이름이 저장 안 된 경우 대비 (수동 지정)
        print("모델에서 피처 이름을 찾을 수 없어 수동 리스트를 사용합니다.")
        manual_features = [
            'jeonse_ratio', 'hug_risk_ratio', 'total_risk_ratio', 'building_age',
            'parking_per_household', 'is_micro_complex', 'estimated_loan_ratio',
            'is_trust_owner', 'short_term_weight',
            'use_아파트', 'use_오피스텔', 'use_다세대주택', 'use_근린생활시설', 'use_기타'
        ]
        df_input = df_input.reindex(columns=manual_features, fill_value=0)

    # 3. 예측 수행
    try:
        prob = rf_model.predict_proba(df_input)[0][1]  # 위험(1)일 확률
    except ValueError as e:
        return {"error": f"모델 예측 오류: 컬럼 불일치 ({e})"}
    is_risky = prob > 0.5

    # 4. 결과 반환
    return {
        "address": address,
        "building_name": bldg_name,
        "deposit": f"{int(deposit_amount / 10000)}만원",
        "risk_score": round(prob * 100, 2),
        "risk_level": "RISKY" if is_risky else "SAFE",
        "details": {
            "hug_ratio": round(input_dict['hug_risk_ratio'] * 100, 1),
            "total_ratio": round(input_dict['total_risk_ratio'] * 100, 1),
            "is_trust": bool(input_dict['is_trust_owner']),
            "is_short_term": bool(input_dict['short_term_weight'] > 0)
        }
    }


# --- 실행 테스트 ---
if __name__ == "__main__":
    # DB에 있는 실제 주소로 테스트해보세요
    test_addr = "인천광역시 부평구 산곡동 145"  # 예시 주소
    test_deposit = 170000000  # 2억 원 (전세 보증금)

    result = predict_risk(test_addr, test_deposit)

    import json

    print(json.dumps(result, indent=4, ensure_ascii=False))