"""
분석 결과 저장 서비스

담당 기능:
- 위험도 분석 결과 DB 저장
- 분석 이력 관리
"""
import logging
from typing import Dict, Any

from sqlalchemy import text

from app.core.database import get_engine
from app.services.address_service import create_address_key

logger = logging.getLogger(__name__)


def save_prediction_result(
        pnu: str,
        building_info_id: int,
        deposit_manwon: float,
        market_price_manwon: float,
        features: Dict[str, Any],
        risk_level: str,
        risk_score: float,
        ai_prob: float
) -> bool:
    """
    예측 결과를 risk_analysis_result 테이블에 저장

    Args:
        pnu: PNU
        building_info_id: 건물 정보 ID
        deposit_manwon: 보증금 (만원)
        market_price_manwon: 시세 (만원)
        features: 분석 피처
        risk_level: 위험 등급
        risk_score: 위험 점수 (0~100)
        ai_prob: AI 예측 확률 (0.0~1.0)

    Returns:
        저장 성공 여부
    """
    engine = get_engine()

    # address_key 생성
    address_key = create_address_key(pnu) if pnu else "UNKNOWN"

    # HUG 한도 계산
    jeonse_ratio = features.get('jeonse_ratio', 0)
    hug_risk_ratio = features.get('hug_risk_ratio', 0)
    total_risk_ratio = features.get('total_risk_ratio', 0)

    # 공시가 기반 HUG 한도 역산 (hug_risk_ratio = deposit / hug_limit)
    hug_safe_limit = deposit_manwon / hug_risk_ratio if hug_risk_ratio > 0 else 0

    params = {
        "building_info_id": building_info_id,
        "address_key": address_key,
        "used_rent_price": deposit_manwon,
        "used_market_price": market_price_manwon,
        "jeonse_ratio": jeonse_ratio,
        "hug_safe_limit": hug_safe_limit,
        "hug_risk_ratio": hug_risk_ratio,
        "total_risk_ratio": total_risk_ratio,
        "estimated_loan_amount": 0,
        "risk_level": risk_level,
        "risk_score": int(risk_score),
        "ai_risk_prob": ai_prob
    }

    # DELETE + INSERT (기존 데이터 갱신)
    sql_delete = text("DELETE FROM risk_analysis_result WHERE address_key = :address_key")

    sql_insert = text("""
        INSERT INTO risk_analysis_result (
            building_info_id, address_key, used_rent_price, used_market_price,
            jeonse_ratio, hug_safe_limit, hug_risk_ratio, total_risk_ratio,
            estimated_loan_amount, risk_level, risk_score, ai_risk_prob, analyzed_at
        ) VALUES (
            :building_info_id, :address_key, :used_rent_price, :used_market_price,
            :jeonse_ratio, :hug_safe_limit, :hug_risk_ratio, :total_risk_ratio,
            :estimated_loan_amount, :risk_level, :risk_score, :ai_risk_prob, NOW()
        )
    """)

    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(sql_delete, {"address_key": address_key})
                conn.execute(sql_insert, params)

        logger.info(f"분석 결과 저장 완료: {address_key}")
        return True

    except Exception as e:
        logger.error(f"분석 결과 저장 실패: {e}")
        return False


def get_previous_analysis(address_key: str) -> Dict[str, Any]:
    """
    이전 분석 결과 조회

    Args:
        address_key: 주소 키

    Returns:
        이전 분석 결과 또는 빈 딕셔너리
    """
    engine = get_engine()

    query = text("""
        SELECT * FROM risk_analysis_result 
        WHERE address_key = :key 
        ORDER BY analyzed_at DESC 
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"key": address_key})
            row = result.mappings().fetchone()

            if row:
                return dict(row)

    except Exception as e:
        logger.error(f"이전 분석 결과 조회 실패: {e}")

    return {}