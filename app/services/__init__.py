"""
Services 패키지

각 서비스 모듈의 주요 함수들을 re-export합니다.
"""

# 메인 예측 함수들
from app.services.predict_service import (
    predict_risk,
    predict_risk_with_ocr
)

# 주소 서비스
from app.services.address_service import (
    normalize_address,
    convert_address_to_pnu,
    parse_pnu,
    get_bjd_map
)

# 건물 서비스
from app.services.building_service import (
    fetch_building_ledger,
    get_building_info_by_pnu,
    check_data_exists_by_pnu
)

# 시세 서비스
from app.services.price_service import (
    estimate_market_price,
    get_trade_price,
    get_public_price,
    calculate_hug_eligibility
)

# 위험도 계산
from app.services.risk_calculator import (
    calculate_risk_features,
    predict_with_model,
    determine_risk_level,
    analyze_risk_factors,
    generate_recommendations
)

# OCR 파서
from app.services.ocr_parser_service import extract_ocr_features

# 결과 저장
from app.services.result_service import save_prediction_result

__all__ = [
    # 메인
    'predict_risk',
    'predict_risk_with_ocr',

    # 주소
    'normalize_address',
    'convert_address_to_pnu',
    'parse_pnu',
    'get_bjd_map',

    # 건물
    'fetch_building_ledger',
    'get_building_info_by_pnu',
    'check_data_exists_by_pnu',

    # 시세
    'estimate_market_price',
    'get_trade_price',
    'get_public_price',
    'calculate_hug_eligibility',

    # 위험도
    'calculate_risk_features',
    'predict_with_model',
    'determine_risk_level',
    'analyze_risk_factors',
    'generate_recommendations',

    # OCR
    'extract_ocr_features',

    # 결과
    'save_prediction_result',
]