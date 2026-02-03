"""
전세사기 위험도 예측 서비스 (메인 오케스트레이터)

이 모듈은 여러 서비스들을 조합하여 예측 기능을 제공합니다.
각 세부 로직은 개별 서비스 모듈에 구현되어 있습니다.

서비스 구조:
- address_service: 주소 정규화, PNU 변환
- building_service: 건축물대장 수집/조회
- price_service: 시세 조회 (실거래가, 공시지가)
- risk_calculator: 위험도 계산, AI 예측
- ocr_parser_service: OCR 데이터 파싱
- result_service: 결과 저장
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from app.core.exceptions import DatabaseConnectionError
from app.services.address_service import (
    normalize_address,
    convert_address_to_pnu,
    parse_pnu
)
from app.services.building_service import (
    fetch_building_ledger,
    get_building_info_by_pnu,
    get_building_info_by_address,
    get_collection_status
)
from app.services.price_service import (
    estimate_market_price,
    get_public_price,
    calculate_hug_eligibility
)
from app.services.risk_calculator import (
    calculate_risk_features,
    predict_with_model,
    determine_risk_level,
    analyze_risk_factors,
    generate_recommendations
)
from app.services.ocr_parser_service import extract_ocr_features
from app.services.result_service import save_prediction_result

# 카카오 API (외부 모듈)
try:
    from app.utils.kakao_localmap_api import get_all_address_and_building_from_kakao
except ImportError:
    get_all_address_and_building_from_kakao = None

logger = logging.getLogger(__name__)


# =============================================================================
# 메인 예측 함수 (단순 버전 - /predict)
# =============================================================================
def predict_risk(address: str, deposit_manwon: int) -> Dict[str, Any]:
    """
    전세사기 위험도 분석 (단순 버전)

    Args:
        address: 주소
        deposit_manwon: 보증금 (만원)

    Returns:
        분석 결과 딕셔너리
    """
    logger.info(f"분석 요청: {address} (보증금: {deposit_manwon:,}만원)")

    try:
        # 1. 주소 정규화 및 변환
        address_info = _resolve_address(address)
        if not address_info:
            return {"error": "주소를 확인할 수 없습니다"}

        lot_addr = address_info["lot_address"]
        road_addr = address_info["road_address"]
        pnu = address_info["pnu"]

        # 2. 건축물대장 수집 (필요시)
        if pnu:
            success, msg = fetch_building_ledger(lot_addr, road_addr, pnu)
            if not success:
                return {"error": f"데이터 수집 실패: {msg}"}

        # 3. 건물 정보 조회
        building_info = _get_building_info(pnu, road_addr, lot_addr)
        if not building_info:
            return {"error": "건물 정보를 찾을 수 없습니다"}

        # 4. 시세 조회
        pnu_for_price = building_info.get('unique_number') or pnu
        area_size = building_info.get('exclusive_area')

        market_price, price_source = estimate_market_price(pnu_for_price, area_size)

        if market_price <= 0:
            return {"error": "시세 정보를 찾을 수 없어 분석할 수 없습니다"}

        # 5. 공시지가 및 HUG 판단
        public_price = get_public_price(pnu_for_price, area_size)
        is_hug_safe, hug_limit, hug_msg = calculate_hug_eligibility(
            public_price, deposit_manwon
        )

        # 6. 위험 피처 계산
        features = calculate_risk_features(
            deposit_manwon=deposit_manwon,
            market_price_manwon=market_price,
            public_price_won=public_price,
            building_info=building_info
        )

        # 7. AI 예측
        prob = predict_with_model(features)
        risk_level = determine_risk_level(prob)
        risk_score = round(prob * 100, 2)

        # 8. 결과 저장
        save_prediction_result(
            pnu=pnu_for_price,
            building_info_id=building_info.get('building_info_id', 0),
            deposit_manwon=deposit_manwon,
            market_price_manwon=market_price,
            features=features,
            risk_level=risk_level,
            risk_score=risk_score,
            ai_prob=prob
        )

        # 9. 응답 생성
        return {
            "address": address,
            "building_name": building_info.get('detail_address', ''),
            "deposit": f"{deposit_manwon:,}만원",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "details": {
                "hug_ratio": round(features['hug_risk_ratio'] * 100, 1),
                "total_ratio": round(features['total_risk_ratio'] * 100, 1),
                "is_trust": bool(features.get('is_trust_owner', 0)),
                "is_short_term": bool(features.get('short_term_weight', 0) > 0)
            }
        }
    except DatabaseConnectionError:
        raise  # DB 연결 오류는 상위로 전파
    except Exception as e:
        logger.exception(f"예측 오류: {e}")
        return {"error": f"분석 중 오류 발생: {str(e)}"}


# =============================================================================
# OCR 기반 예측 함수 (/predict/v2)
# =============================================================================
def predict_risk_with_ocr(
        address: str,
        deposit_manwon: int,
        ocr_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    OCR 데이터 기반 정밀 위험도 분석

    Args:
        address: 주소
        deposit_manwon: 보증금 (만원)
        ocr_data: OCR 추출 데이터

    Returns:
        구조화된 API 응답
    """
    logger.info("=" * 60)
    logger.info(f"🕵️ 예측 시작: {address}")
    logger.info(f"💰 보증금: {deposit_manwon:,}만원")
    logger.info("-" * 60)

    try:
        # 1. OCR 피처 추출
        ocr_features = extract_ocr_features(ocr_data)
        pnu = ocr_features.get('unique_number', '')
        area_size = ocr_features.get('area_size')

        # 2. 시세 조회
        market_price, price_source = estimate_market_price(pnu, area_size)

        # 3. 공시지가 및 HUG 판단
        public_price = get_public_price(pnu, area_size)
        is_hug_safe, hug_limit, hug_msg = calculate_hug_eligibility(
            public_price, deposit_manwon
        )

        # 시세가 없으면 HUG 한도를 시세로 사용
        if market_price <= 0 and hug_limit > 0:
            market_price = hug_limit
            price_source = "Public_Price_Estimate"

        if market_price <= 0:
            return _error_response(422, "시세 데이터 없음",
                                   "공시지가 또는 실거래가 데이터를 찾을 수 없어 분석할 수 없습니다")

        # 4. 위험 피처 계산
        features = calculate_risk_features(
            deposit_manwon=deposit_manwon,
            market_price_manwon=market_price,
            public_price_won=public_price,
            ocr_features=ocr_features
        )

        # HUG 위험 비율 업데이트
        if public_price > 0:
            features['hug_risk_ratio'] = (deposit_manwon * 10000) / (public_price * 1.26)

        # 5. AI 예측
        prob = predict_with_model(features)
        risk_level = determine_risk_level(prob)

        # 6. 위험 요인 분석
        risk_factors = analyze_risk_factors(features, is_hug_safe)

        # 7. 권장사항 생성
        recommendations = generate_recommendations(
            risk_level=risk_level,
            is_hug_safe=is_hug_safe,
            jeonse_ratio=features.get('jeonse_ratio', 0)
        )

        # 8. HUG 결과 구조화
        actual_coverage = min(hug_limit, deposit_manwon)
        hug_result = {
            "is_eligible": is_hug_safe,
            "safe_limit": int(hug_limit * 10000),
            "coverage_ratio": round((actual_coverage / deposit_manwon * 100), 1) if deposit_manwon > 0 else 0,
            "message": hug_msg
        }

        if not is_hug_safe:
            jeonse_ratio = features.get('jeonse_ratio', 0)
            if jeonse_ratio > 0.8:
                hug_result["reason"] = f"전세가율 {jeonse_ratio * 100:.1f}% 초과 (기준: 80%)"
            else:
                hug_result["reason"] = "기타 심사 기준 미달"

        # 9. 응답 생성
        return {
            "meta": {
                "code": 200,
                "message": "전세사기 위험도 분석 완료",
                "timestamp": datetime.now().isoformat()
            },
            "data": {
                "address": address,
                "deposit": int(deposit_manwon * 10000),
                "market_price": int(market_price * 10000),
                "price_source": price_source,

                "risk_score": round(prob * 100, 1),
                "risk_level": risk_level,
                "major_risk_factors": risk_factors,

                "hug_result": hug_result,

                "details": {
                    "jeonse_ratio": round(features.get('jeonse_ratio', 0) * 100, 1),
                    "senior_debt": int(ocr_features.get('real_debt_manwon', 0) * 10000),
                    "is_illegal_building": bool(ocr_features.get('is_illegal', 0)),
                    "is_trust": bool(ocr_features.get('is_trust_owner', 0)),
                    "building_age": round(features.get('building_age', 0), 1),
                    "ownership_duration_months": ocr_features.get('ownership_duration_months')
                },

                "recommendations": recommendations
            },
            "_debug_info": {
                "pnu": pnu,
                "features_used": {k: v for k, v in features.items()
                                  if k not in ['estimated_loan_ratio', 'parking_per_household']}
            }
        }
    except DatabaseConnectionError:
        raise  # DB 연결 오류는 상위로 전파
    except Exception as e:
        logger.exception(f"예측 오류: {e}")
        return _error_response(500, "서버 오류가 발생했습니다", f"분석 중 오류 발생: {str(e)}")


# =============================================================================
# 헬퍼 함수들
# =============================================================================
def _resolve_address(address: str) -> Optional[Dict[str, str]]:
    """
    주소 정규화 및 PNU 변환

    Returns:
        {"lot_address": ..., "road_address": ..., "pnu": ...} 또는 None
    """
    # 카카오 API로 주소 정규화
    if get_all_address_and_building_from_kakao:
        try:
            lot_addr, road_addr, building_name = get_all_address_and_building_from_kakao(address)
            lot_addr = normalize_address(lot_addr)
            road_addr = normalize_address(road_addr)
        except Exception as e:
            logger.warning(f"카카오 API 오류: {e}")
            lot_addr = normalize_address(address)
            road_addr = lot_addr
    else:
        lot_addr = normalize_address(address)
        road_addr = lot_addr

    # PNU 변환
    pnu, msg = convert_address_to_pnu(lot_addr)

    return {
        "lot_address": lot_addr,
        "road_address": road_addr,
        "pnu": pnu
    }


def _get_building_info(
        pnu: Optional[str],
        road_addr: str,
        lot_addr: str
) -> Optional[Dict[str, Any]]:
    """건물 정보 조회 (PNU 우선, 없으면 주소로)"""
    if pnu:
        info = get_building_info_by_pnu(pnu)
        if info:
            return info

    return get_building_info_by_address(road_addr, lot_addr)


def _error_response(code: int, message: str, detail: str) -> Dict[str, Any]:
    """에러 응답 생성"""
    return {
        "meta": {
            "code": code,
            "message": message,
            "timestamp": datetime.now().isoformat()
        },
        "errors": [
            {"field": "prediction", "message": detail}
        ]
    }


# =============================================================================
# 테스트용
# =============================================================================
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    # 테스트
    test_addr = "인천광역시 부평구 산곡동 145"
    result = predict_risk(test_addr, 20000)
    print(result)