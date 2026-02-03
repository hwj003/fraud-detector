"""
건축물대장 데이터 수집 및 조회 서비스

담당 기능:
- 전유부/표제부 API 수집
- 건물 정보 DB 조회
- 데이터 존재 여부 확인
"""
import logging
from typing import Tuple, Optional, Dict, Any

from sqlalchemy import text

from app.core.database import get_engine
from app.services.address_service import normalize_address

logger = logging.getLogger(__name__)


# =============================================================================
# 외부 API 모듈 import (지연 로딩)
# =============================================================================
def _get_api_modules():
    """API 관련 모듈 지연 로딩 (순환 참조 방지)"""
    from scripts.fetch_data.fetch_ledger_exclusive import get_access_token, fetch_target_middle_unit
    from scripts.fetch_data.fetch_ledger_title import collect_title_data
    from app.utils.kakao_localmap_api import (
        get_road_address_from_kakao,
        get_building_name_from_kakao
    )

    return {
        'get_access_token': get_access_token,
        'fetch_target_middle_unit': fetch_target_middle_unit,
        'collect_title_data': collect_title_data,
        'get_road_address_from_kakao': get_road_address_from_kakao,
        'get_building_name_from_kakao': get_building_name_from_kakao
    }


# =============================================================================
# 데이터 존재 여부 확인
# =============================================================================
def check_data_exists_by_pnu(table_name: str, pnu_prefix: str) -> bool:
    """
    PNU 기준으로 테이블에 데이터 존재 여부 확인

    Args:
        table_name: 테이블명 (building_info, building_title_info)
        pnu_prefix: PNU 앞부분

    Returns:
        데이터 존재 여부
    """
    if not pnu_prefix:
        return False

    engine = get_engine()

    query = text(f"""
        SELECT 1 FROM {table_name} 
        WHERE unique_number LIKE :pnu_pattern 
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"pnu_pattern": f"{pnu_prefix}%"})
            return result.fetchone() is not None
    except Exception as e:
        logger.error(f"데이터 존재 확인 실패 ({table_name}): {e}")
        return False


def get_collection_status(pnu: str) -> Dict[str, bool]:
    """
    해당 PNU의 전유부/표제부 수집 상태 확인

    Returns:
        {"exclusive": bool, "title": bool}
    """
    return {
        "exclusive": check_data_exists_by_pnu("building_info", pnu),
        "title": check_data_exists_by_pnu("building_title_info", pnu)
    }


# =============================================================================
# 전유부/표제부 수집
# =============================================================================
def _collect_exclusive_with_retry(token: str, address: str) -> bool:
    """
    전유부 수집 (지번 -> 도로명 재시도)

    Returns:
        수집 성공 여부
    """
    api = _get_api_modules()

    logger.info(f"전유부(Exclusive) 수집 시작: {address}")

    # 1차 시도: 지번 주소
    if api['fetch_target_middle_unit'](token, address, address):
        return True

    # 2차 시도: 도로명 + 건물명
    try:
        road_part = api['get_road_address_from_kakao'](address)
        build_part = api['get_building_name_from_kakao'](address)
        retry_address = f"{road_part} {build_part}".strip()

        logger.info(f"전유부 재시도 (도로명): {retry_address}")

        if api['fetch_target_middle_unit'](token, retry_address, address):
            return True

    except Exception as e:
        logger.error(f"전유부 재시도 주소 생성 실패: {e}")

    return False


def _collect_title_with_retry(token: str, address: str) -> bool:
    """
    표제부 수집 (지번 -> 도로명 재시도)

    Returns:
        수집 성공 여부
    """
    api = _get_api_modules()

    logger.info(f"표제부(Title) 수집 시작: {address}")

    # 1차 시도: 지번 주소
    if api['collect_title_data'](token, address, address):
        return True

    # 2차 시도: 도로명 + 건물명
    try:
        road_part = api['get_road_address_from_kakao'](address)
        build_part = api['get_building_name_from_kakao'](address)
        retry_address = f"{road_part} {build_part}".strip()

        logger.info(f"표제부 재시도 (도로명): {retry_address}")

        if api['collect_title_data'](token, retry_address, address):
            return True

    except Exception as e:
        logger.error(f"표제부 재시도 주소 생성 실패: {e}")

    return False


def fetch_building_ledger(
        address: str,
        road_addr: str,
        target_pnu: str
) -> Tuple[bool, str]:
    """
    건축물대장(전유부/표제부) 수집 메인 함수

    Args:
        address: 지번 주소
        road_addr: 도로명 주소
        target_pnu: PNU

    Returns:
        Tuple[성공여부, 메시지]
    """
    api = _get_api_modules()

    logger.info(f"수집 상태 점검: {address}")

    # 1. 주소 정규화
    address = normalize_address(address)
    road_addr = normalize_address(road_addr)

    # 2. 기존 데이터 확인
    status = get_collection_status(target_pnu)

    if status["exclusive"] and status["title"]:
        logger.info("전유부/표제부 모두 이미 수집됨")
        return True, "이미 수집됨"

    # 3. API 토큰 발급
    token = api['get_access_token']()
    if not token:
        return False, "API 토큰 발급 실패"

    # 4. 전유부 수집
    if not status["exclusive"]:
        if not _collect_exclusive_with_retry(token, address):
            return False, "전유부 수집 실패 (데이터 없음)"
    else:
        logger.info("전유부 데이터 보유 중 (스킵)")

    # 5. 표제부 수집
    if not status["title"]:
        if not _collect_title_with_retry(token, address):
            return False, "표제부 수집 실패 (데이터 없음)"
    else:
        logger.info("표제부 데이터 보유 중 (스킵)")

    return True, "수집 완료"


# =============================================================================
# 건물 정보 조회
# =============================================================================
def get_building_info_by_pnu(pnu: str) -> Optional[Dict[str, Any]]:
    """
    PNU로 건물 정보 조회 (전유부 + 표제부 + 공시가 조인)

    Args:
        pnu: PNU 문자열

    Returns:
        건물 정보 딕셔너리 또는 None
    """
    engine = get_engine()

    query = text("""
        SELECT 
            b.id as building_info_id,
            b.unique_number, 
            b.detail_address, 
            b.main_use, 
            b.exclusive_area,
            b.owner_name, 
            b.ownership_changed_date, 
            b.is_violating_building,
            p.price as public_price,
            t.household_cnt, 
            t.parking_cnt, 
            t.elevator_cnt, 
            t.use_apr_day
        FROM building_info b
        LEFT JOIN public_price_history p ON b.id = p.building_info_id
        LEFT JOIN building_title_info t 
            ON SUBSTRING(b.unique_number, 1, 14) = SUBSTRING(t.unique_number, 1, 14)
        WHERE b.unique_number LIKE :pnu_pattern
        ORDER BY p.base_date DESC 
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"pnu_pattern": f"{pnu}%"})
            row = result.mappings().fetchone()

            if row:
                return dict(row)
            return None

    except Exception as e:
        logger.error(f"건물 정보 조회 실패: {e}")
        return None


def get_building_info_by_address(
        road_addr: str,
        lot_addr: str
) -> Optional[Dict[str, Any]]:
    """
    주소로 건물 정보 조회 (PNU가 없을 때 Fallback)

    Args:
        road_addr: 도로명 주소
        lot_addr: 지번 주소

    Returns:
        건물 정보 딕셔너리 또는 None
    """
    engine = get_engine()

    query = text("""
        SELECT 
            b.id as building_info_id,
            b.unique_number, 
            b.detail_address, 
            b.main_use, 
            b.exclusive_area,
            b.owner_name, 
            b.ownership_changed_date, 
            b.is_violating_building,
            p.price as public_price,
            t.household_cnt, 
            t.parking_cnt, 
            t.elevator_cnt, 
            t.use_apr_day
        FROM building_info b
        LEFT JOIN public_price_history p ON b.id = p.building_info_id
        LEFT JOIN building_title_info t 
            ON SUBSTRING(b.unique_number, 1, 14) = SUBSTRING(t.unique_number, 1, 14)
        WHERE b.road_address LIKE :road_pattern 
           OR b.lot_address LIKE :lot_pattern
        ORDER BY p.base_date DESC 
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {
                "road_pattern": f"%{road_addr}%",
                "lot_pattern": f"%{lot_addr}%"
            })
            row = result.mappings().fetchone()

            if row:
                return dict(row)
            return None

    except Exception as e:
        logger.error(f"주소 기반 건물 정보 조회 실패: {e}")
        return None