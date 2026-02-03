import numpy as np
import pandas as pd
from datetime import datetime

# 학습 때 사용한 피처 순서와 동일하게 유지
TRAIN_FEATURES = [
    'jeonse_ratio', 'hug_risk_ratio', 'total_risk_ratio', 'estimated_loan_ratio',
    'building_age', 'is_illegal', 'parking_per_household', 'is_micro_complex',
    'is_trust_owner', 'short_term_weight',
    'type_APT', 'type_OFFICETEL', 'type_VILLA', 'type_ETC'
]


def calculate_risk_features(
        deposit_amount: float,  # 보증금 (만원)
        market_price: float,  # 시세 (만원)
        real_debt: float,  # 등기부상 채권총액 (만원)
        main_use: str,  # 주용도 (아파트, 다세대 등)
        usage_approval_date: str,  # 사용승인일 ('YYYY-MM-DD')
        is_illegal: int,  # 위반건축물 여부 (0 or 1)
        is_trust_owner: int,  # 신탁 소유 여부 (0 or 1)
        short_term_weight: float,  # 단기 소유 가중치 (0.0 ~ 0.3)
        parking_count: int = 0,    # 총 주차대수
        household_count: int = 0,  # 총 세대수
) -> dict:
    """
    원천 데이터를 받아 AI 모델 입력용 피처 딕셔너리로 변환하는 함수
    """

    # 1. 시세가 0이거나 없을 경우 방어 로직 (보증금 역산)
    if market_price <= 0:
        market_price = deposit_amount * 1.25  # 전세가율 80% 가정

    # 2. 비율 지표 계산
    # (1) 전세가율 (순수 보증금 / 시세)
    jeonse_ratio = deposit_amount / market_price

    # (2) 깡통전세율 ( (보증금 + 선순위채권) / 시세 ) -> *핵심*
    total_risk_ratio = (deposit_amount + real_debt) / market_price

    # (3) HUG 위험도 (공시가는 시세의 70%로 추정)
    est_public_price = market_price * 0.7
    hug_limit = est_public_price * 1.26
    hug_risk_ratio = deposit_amount / hug_limit if hug_limit > 0 else 0

    # 3. 정성적 위험 점수 (estimated_loan_ratio)
    # 건물 유형 가중치
    type_w = 0.0
    main_use_str = str(main_use)
    if '근린' in main_use_str:
        type_w = 0.4
    elif any(x in main_use_str for x in ['다세대', '연립', '오피스텔', '빌라']):
        type_w = 0.1

    # 점수 합산 (0 ~ 1.0 범위로 클리핑)
    risk_score_val = np.clip(type_w + short_term_weight + (0.5 if is_trust_owner else 0), 0, 1.0)

    # 4. 건물 연식 계산
    building_age = 0
    try:
        if usage_approval_date:
            # 1. 입력값 문자열 변환 및 정리
            raw_date = str(usage_approval_date).strip()

            # 2. 온점(.)이나 슬래시(/)를 하이픈(-)으로 통일하고, 끝에 붙은 특수문자 제거
            # 예: "2001.5.3." -> "2001-5-3"
            clean_date = raw_date.replace('.', '-').replace('/', '-').strip('-')

            # 3. Pandas의 강력한 파서 사용 (알아서 YYYY-MM-DD 등을 해석해줌)
            dt = pd.to_datetime(clean_date, errors='coerce')

            if pd.notnull(dt):
                building_age = (datetime.now() - dt).days / 365.25
            else:
                # 변환 실패 시 (예: "없음" 등)
                building_age = 10
    except:
        building_age = 10  # 파싱 실패 시 기본값

    # 5. One-Hot Encoding (건물 유형)
    type_dict = {'type_APT': 0, 'type_OFFICETEL': 0, 'type_VILLA': 0, 'type_ETC': 0}

    if '아파트' in main_use_str:
        type_dict['type_APT'] = 1
    elif '오피스텔' in main_use_str:
        type_dict['type_OFFICETEL'] = 1
    elif any(x in main_use_str for x in ['다세대', '연립', '빌라']):
        type_dict['type_VILLA'] = 1
    else:
        type_dict['type_ETC'] = 1

    # 피처 계산 로직 활성화
    parking_per_household = parking_count / household_count if household_count > 0 else 0

    # 100세대 미만이면 나홀로/소규모 단지로 판단 (위험 가중)
    is_micro_complex = 1 if household_count < 100 else 0

    # 6. 최종 딕셔너리 조립
    features = {
        'jeonse_ratio': jeonse_ratio,
        'hug_risk_ratio': hug_risk_ratio,
        'total_risk_ratio': total_risk_ratio,
        'estimated_loan_ratio': risk_score_val,
        'building_age': building_age,
        'is_illegal': is_illegal,
        'parking_per_household': parking_per_household,  # (입력값 없으면 기본 0)
        'is_micro_complex': is_micro_complex,  # (입력값 없으면 나홀로 1)
        'is_trust_owner': is_trust_owner,
        'short_term_weight': short_term_weight,
    }
    features.update(type_dict)  # 원핫 인코딩 합치기

    return features


def convert_to_model_input(features: dict) -> pd.DataFrame:
    """
    피처 딕셔너리를 받아서 모델 예측용 DataFrame으로 변환 (컬럼 순서 보장)
    """
    df = pd.DataFrame([features])

    # 학습 때 썼던 컬럼이 없으면 0으로 채우고, 순서 맞춤
    for col in TRAIN_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[TRAIN_FEATURES]