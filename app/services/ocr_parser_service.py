"""
OCR 데이터 파싱 서비스

담당 기능:
- 건축물대장 OCR 결과 파싱
- 등기부등본 OCR 결과 파싱
- 권리 정보 추출
"""
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def extract_ocr_features(ocr_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    OCR 데이터에서 분석에 필요한 피처 추출

    Args:
        ocr_data: {
            "ledger": {...},    # 건축물대장 OCR 결과
            "registry": {...}   # 등기부등본 OCR 결과
        }

    Returns:
        추출된 피처 딕셔너리
    """
    ledger = ocr_data.get('ledger') or {}
    registry = ocr_data.get('registry') or {}

    # 건축물대장 정보 추출
    ledger_features = _parse_building_ledger(ledger)

    # 등기부등본 정보 추출
    registry_features = _parse_registry(registry)

    # 결합
    return {
        **ledger_features,
        **registry_features
    }


def _parse_building_ledger(ledger: Dict) -> Dict[str, Any]:
    """
    건축물대장 OCR 결과 파싱

    Returns:
        {
            "unique_number": PNU,
            "main_use": 주용도,
            "area_size": 전용면적,
            "usage_approval_date": 사용승인일,
            "is_illegal": 위반건축물 여부
        }
    """
    building_status = ledger.get('building_status', {})
    document_info = ledger.get('document_info', {})
    safety = ledger.get('safety_check', {})

    # 기본 정보
    unique_number = document_info.get('unique_number', '')
    main_use = str(building_status.get('main_usage', ''))
    usage_approval_date = building_status.get('usage_approval_date', '')

    # 면적 파싱
    area_size = None
    area_raw = building_status.get('area')
    if area_raw:
        try:
            # "84.5㎡" 같은 형식에서 숫자 추출
            nums = re.findall(r'[\d.]+', str(area_raw))
            if nums:
                area_size = float(nums[0])
        except:
            pass

    # 위반 건축물 여부
    is_illegal = 1 if safety.get('is_violator') else 0

    return {
        "unique_number": unique_number,
        "main_use": main_use,
        "area_size": area_size,
        "usage_approval_date": usage_approval_date,
        "is_illegal": is_illegal
    }


def _parse_registry(registry: Dict) -> Dict[str, Any]:
    """
    등기부등본 OCR 결과 파싱

    Returns:
        {
            "is_trust_owner": 신탁 여부,
            "short_term_weight": 단기소유 가중치,
            "real_debt_manwon": 선순위채권 (만원),
            "ownership_duration_months": 소유기간 (개월)
        }
    """
    basic_info = registry.get('basic_info', {})
    risk_factors = registry.get('risk_factors', {})

    # 1. 신탁 여부
    owner_name = basic_info.get('owner', '')
    trust_content = risk_factors.get('trust_content', '')
    is_trust_owner = 1 if ('신탁' in str(owner_name) or '신탁' in str(trust_content)) else 0

    # 2. 소유 기간 및 단기 소유 가중치
    short_term_weight = 0.0
    ownership_duration_months = None

    ownership_date_str = basic_info.get('ownership_date', '')
    if ownership_date_str:
        parsed_date = _parse_date_string(ownership_date_str)
        if parsed_date:
            days_diff = (datetime.now() - parsed_date).days
            ownership_duration_months = days_diff // 30

            if days_diff < 90:
                short_term_weight = 0.3  # 3개월 미만: 높은 위험
            elif days_diff < 730:
                short_term_weight = 0.1  # 2년 미만: 약간 위험

    # 3. 선순위 채권 합산
    real_debt_won = _calculate_total_debt(registry.get('debts', []))
    real_debt_manwon = real_debt_won / 10000

    return {
        "is_trust_owner": is_trust_owner,
        "short_term_weight": short_term_weight,
        "real_debt_manwon": real_debt_manwon,
        "ownership_duration_months": ownership_duration_months
    }


def _parse_date_string(date_str: str) -> Optional[datetime]:
    """
    다양한 형식의 날짜 문자열 파싱

    지원 형식:
    - 20240101
    - 2024-01-01
    - 2024.01.01
    """
    if not date_str:
        return None

    # 숫자만 추출
    date_str_clean = str(date_str).replace('-', '').replace('.', '').replace(' ', '')

    # 8자리 날짜 찾기
    dates = re.findall(r'\d{8}', date_str_clean)

    if dates:
        try:
            return datetime.strptime(dates[0], '%Y%m%d')
        except ValueError:
            pass

    return None


def _calculate_total_debt(debts: List[Dict]) -> float:
    """
    채권 목록에서 유효한 선순위 채권 합산

    Args:
        debts: 채권 목록 [{"amount": "100,000,000", "status": "active"}, ...]

    Returns:
        총 채권액 (원)
    """
    total = 0.0

    for item in debts:
        # 말소된 채권 제외
        status = item.get('status', '').lower()
        if status in ['cancelled', 'deleted', '말소']:
            continue

        # 금액 파싱
        try:
            raw_amount = str(item.get('amount', '0'))
            # 쉼표 제거 후 숫자만 추출
            nums = re.findall(r'\d+', raw_amount.replace(',', ''))
            if nums:
                total += float(''.join(nums))
        except:
            continue

    return total


def extract_address_from_ocr(ocr_data: Dict) -> Optional[str]:
    """
    OCR 데이터에서 주소 추출

    Args:
        ocr_data: OCR 결과

    Returns:
        주소 문자열 또는 None
    """
    # 건축물대장에서 추출
    ledger = ocr_data.get('ledger', {})
    building_status = ledger.get('building_status', {})

    address = building_status.get('address') or building_status.get('lot_address')
    if address:
        return str(address)

    # 등기부등본에서 추출
    registry = ocr_data.get('registry', {})
    basic_info = registry.get('basic_info', {})

    address = basic_info.get('address') or basic_info.get('property_address')
    if address:
        return str(address)

    return None