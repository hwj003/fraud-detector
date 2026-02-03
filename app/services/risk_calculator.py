"""
위험도 계산 서비스

담당 기능:
- 위험 피처 계산
- AI 모델 예측
- 위험 등급 판정
- 위험 요인 분석
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# 프로젝트 루트
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 건물 유형 컬럼
USE_COLS = ['type_APT', 'type_OFFICETEL', 'type_VILLA', 'type_ETC']

# =============================================================================
# AI 모델 로더 (Lazy Loading)
# =============================================================================
_rf_model = None


def get_model():
    """Random Forest 모델 로드 (싱글톤)"""
    global _rf_model

    if _rf_model is not None:
        return _rf_model

    model_path = os.path.join(PROJECT_ROOT, 'models', 'fraud_rf_model.pkl')

    if not os.path.exists(model_path):
        logger.warning(f"모델 파일 없음: {model_path}")
        return None

    try:
        _rf_model = joblib.load(model_path)
        logger.info(f"모델 로드 성공: {model_path}")
        return _rf_model
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None


# =============================================================================
# 위험 피처 계산
# =============================================================================
def calculate_risk_features(
        deposit_manwon: float,
        market_price_manwon: float,
        public_price_won: float = 0,
        building_info: Optional[Dict] = None,
        ocr_features: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    위험도 분석용 피처 계산

    Args:
        deposit_manwon: 보증금 (만원)
        market_price_manwon: 시세 (만원)
        public_price_won: 공시가 (원)
        building_info: 건물 정보 (DB 조회 결과)
        ocr_features: OCR 추출 피처

    Returns:
        피처 딕셔너리
    """
    building_info = building_info or {}
    ocr_features = ocr_features or {}

    # 1. 전세가율
    jeonse_ratio = deposit_manwon / market_price_manwon if market_price_manwon > 0 else 0

    # 2. HUG 위험 비율
    hug_limit = (public_price_won * 1.26 / 10000) if public_price_won > 0 else market_price_manwon
    hug_risk_ratio = deposit_manwon / hug_limit if hug_limit > 0 else 0

    # 3. 선순위 채권 반영 총 위험 비율
    real_debt = ocr_features.get('real_debt_manwon', 0)
    total_risk_ratio = (deposit_manwon + real_debt) / market_price_manwon if market_price_manwon > 0 else 0

    # 4. 신탁 여부
    owner_name = str(building_info.get('owner_name', ''))
    is_trust = ocr_features.get('is_trust_owner', 0) or (1 if '신탁' in owner_name else 0)

    # 5. 단기 소유 가중치
    short_term_weight = ocr_features.get('short_term_weight', 0)
    if short_term_weight == 0 and building_info.get('ownership_changed_date'):
        try:
            own_date = pd.to_datetime(building_info['ownership_changed_date'])
            days_diff = (datetime.now() - own_date).days

            if days_diff < 90:
                short_term_weight = 0.3
            elif days_diff < 730:
                short_term_weight = 0.1
        except:
            pass

    # 6. 건물 유형 가중치
    main_use = str(building_info.get('main_use', '') or ocr_features.get('main_use', ''))
    type_weight = 0.0

    if '근린' in main_use:
        type_weight = 0.4
    elif any(t in main_use for t in ['다세대', '오피스텔', '연립']):
        type_weight = 0.1

    # 7. 위험 점수 (정성적 요소)
    risk_score_val = np.clip(type_weight + short_term_weight + (0.5 if is_trust else 0), 0, 1.0)

    # 8. 건물 나이
    building_age = 10  # 기본값
    use_apr_day = building_info.get('use_apr_day') or ocr_features.get('usage_approval_date')

    if use_apr_day:
        try:
            apr_date = pd.to_datetime(use_apr_day)
            building_age = (datetime.now() - apr_date).days / 365.25
        except:
            pass

    # 9. 기타 피처
    household_cnt = building_info.get('household_cnt') or 1
    if household_cnt < 1:
        household_cnt = 1

    parking_per_household = (building_info.get('parking_cnt') or 0) / household_cnt
    is_micro_complex = 1 if household_cnt < 100 else 0
    is_illegal = ocr_features.get('is_illegal', 0) or (
        1 if str(building_info.get('is_violating_building', '')).strip() == 'Y' else 0
    )

    # 10. One-Hot Encoding (건물 유형)
    type_dict = {col: 0 for col in USE_COLS}

    if '아파트' in main_use:
        type_dict['type_APT'] = 1
    elif '오피스텔' in main_use:
        type_dict['type_OFFICETEL'] = 1
    elif '다세대' in main_use or '연립' in main_use:
        type_dict['type_VILLA'] = 1
    else:
        type_dict['type_ETC'] = 1

    # 결과 조합
    features = {
        'jeonse_ratio': jeonse_ratio,
        'hug_risk_ratio': hug_risk_ratio,
        'total_risk_ratio': total_risk_ratio,
        'estimated_loan_ratio': risk_score_val,
        'building_age': building_age,
        'is_illegal': is_illegal,
        'parking_per_household': parking_per_household,
        'is_micro_complex': is_micro_complex,
        'is_trust_owner': is_trust,
        'short_term_weight': short_term_weight,
        'real_debt': real_debt,
        **type_dict
    }

    return features


# =============================================================================
# AI 모델 예측
# =============================================================================
def predict_with_model(features: Dict[str, Any]) -> float:
    """
    AI 모델로 위험 확률 예측

    Args:
        features: 피처 딕셔너리

    Returns:
        위험 확률 (0.0 ~ 1.0)
    """
    model = get_model()

    if model is None:
        # 모델 없으면 룰 베이스로 대체
        return _rule_based_prediction(features)

    try:
        # DataFrame 변환
        df_input = pd.DataFrame([features])

        # 모델 피처 순서 맞추기
        try:
            train_features = model.feature_names_in_
            df_input = df_input.reindex(columns=train_features, fill_value=0)
        except AttributeError:
            logger.warning("모델 피처 이름 확인 불가")

        # 예측
        prob = model.predict_proba(df_input)[0][1]
        return float(prob)

    except Exception as e:
        logger.error(f"모델 예측 오류: {e}")
        return _rule_based_prediction(features)


def _rule_based_prediction(features: Dict[str, Any]) -> float:
    """룰 베이스 위험도 예측 (모델 없을 때 Fallback)"""
    prob = 0.0

    # 전세가율 80% 초과
    if features.get('jeonse_ratio', 0) >= 0.8:
        prob = max(prob, 0.8)
    elif features.get('jeonse_ratio', 0) >= 0.7:
        prob = max(prob, 0.5)

    # 총 위험 비율 (선순위 채권 포함)
    if features.get('total_risk_ratio', 0) >= 0.9:
        prob = max(prob, 0.9)
    elif features.get('total_risk_ratio', 0) >= 0.8:
        prob = max(prob, 0.7)

    # 신탁 + 단기 소유
    if features.get('is_trust_owner', 0) and features.get('short_term_weight', 0) > 0:
        prob = max(prob, 0.6)

    return prob


def determine_risk_level(prob: float) -> str:
    """
    확률을 위험 등급으로 변환

    Args:
        prob: 위험 확률 (0.0 ~ 1.0)

    Returns:
        "SAFE" | "CAUTION" | "RISKY"
    """
    if prob < 0.4:
        return "SAFE"
    elif prob < 0.7:
        return "CAUTION"
    else:
        return "RISKY"


# =============================================================================
# 위험 요인 분석
# =============================================================================
def analyze_risk_factors(
        features: Dict[str, Any],
        is_hug_safe: bool
) -> List[Dict[str, str]]:
    """
    위험 요인을 구조화된 형태로 분석

    Args:
        features: 피처 딕셔너리
        is_hug_safe: HUG 가입 가능 여부

    Returns:
        위험 요인 리스트 [{"type": ..., "severity": ..., "message": ...}, ...]
    """
    risk_factors = []

    # 1. HUG 불가
    if not is_hug_safe:
        risk_factors.append({
            "type": "HUG_INELIGIBLE",
            "severity": "CRITICAL",
            "message": "HUG 전세보증금 반환보증 가입 불가"
        })

    # 2. 높은 전세가율
    jeonse_ratio = features.get('jeonse_ratio', 0)
    if jeonse_ratio > 0.8:
        risk_factors.append({
            "type": "HIGH_LTV",
            "severity": "HIGH",
            "message": f"전세가율이 {jeonse_ratio * 100:.1f}%로 매우 높음 (깡통전세 위험)"
        })
    elif jeonse_ratio > 0.7:
        risk_factors.append({
            "type": "HIGH_LTV",
            "severity": "MEDIUM",
            "message": f"전세가율이 {jeonse_ratio * 100:.1f}%로 다소 높음"
        })

    # 3. 선순위 채권
    real_debt = features.get('real_debt', 0)
    if real_debt > 0:
        risk_factors.append({
            "type": "SENIOR_DEBT",
            "severity": "HIGH",
            "message": f"선순위 채권 {real_debt:,.0f}만원 존재 (보증금 회수 우선순위 낮음)"
        })

    # 4. 위반 건축물
    if features.get('is_illegal', 0):
        risk_factors.append({
            "type": "ILLEGAL_BUILDING",
            "severity": "HIGH",
            "message": "위반 건축물로 등재됨 (법적 제재 가능)"
        })

    # 5. 신탁 부동산
    if features.get('is_trust_owner', 0):
        risk_factors.append({
            "type": "TRUST_PROPERTY",
            "severity": "MEDIUM",
            "message": "신탁 부동산으로 권리 관계가 복잡할 수 있음"
        })

    # 6. 단기 소유
    if features.get('short_term_weight', 0) > 0:
        severity = "HIGH" if features['short_term_weight'] >= 0.3 else "MEDIUM"
        risk_factors.append({
            "type": "SHORT_OWNERSHIP",
            "severity": severity,
            "message": "건물 소유 기간이 짧음 (투기 의심)"
        })

    # 7. 노후 건물
    building_age = features.get('building_age', 0)
    if building_age > 30:
        risk_factors.append({
            "type": "OLD_BUILDING",
            "severity": "LOW",
            "message": f"건물 연식 {building_age:.0f}년으로 노후화됨"
        })

    # 위험 요인 없음
    if not risk_factors:
        risk_factors.append({
            "type": "NONE",
            "severity": "LOW",
            "message": "특이한 위험 요인이 발견되지 않았습니다"
        })

    return risk_factors


# =============================================================================
# 권장사항 생성
# =============================================================================
def generate_recommendations(
        risk_level: str,
        is_hug_safe: bool,
        jeonse_ratio: float
) -> List[str]:
    """
    사용자 권장 조치사항 생성

    Args:
        risk_level: 위험 등급
        is_hug_safe: HUG 가입 가능 여부
        jeonse_ratio: 전세가율

    Returns:
        권장사항 리스트
    """
    recommendations = []

    # HUG 관련
    if is_hug_safe:
        recommendations.append("HUG 보증보험 가입을 권장합니다")
    else:
        recommendations.append("HUG 보증보험 가입이 불가하므로 계약 재검토를 권장합니다")

    # 위험 등급 관련
    if risk_level in ["CAUTION", "RISKY"]:
        recommendations.append("등기부등본 재확인 권장 (최근 3개월 이내)")
        recommendations.append("임대인의 재정 상태 및 다른 채무 여부 확인 필요")

    # 전세가율 관련
    if jeonse_ratio > 0.75:
        recommendations.append("전세가율이 높으므로 월세 전환 또는 보증금 인하 협상 검토")

    # 공통
    recommendations.append("계약 전 법무사 자문을 통한 권리 관계 검토 권장")

    return recommendations