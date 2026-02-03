"""
시세 조회 서비스

담당 기능:
- 실거래가(매매/전세) 조회
- 공시지가 조회
- API 수집 이력 관리
- 시세 추정 로직
"""
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.core import get_engine, is_db_available
from app.core.exceptions import DatabaseConnectionError
from app.services.address_service import parse_pnu, pnu_to_raw_format

logger = logging.getLogger(__name__)

# 재시도 설정
MAX_DB_RETRIES=2


def _execute_query_safe(query, params: dict = None, operation_name: str = "DB 작업"):
    """
    안전한 쿼리 실행 (연결 실패 시 예외 발생)

    Args:
        query: SQLAlchemy text 쿼리
        params: 쿼리 파라미터
        operation_name: 작업명 (로깅용)

    Returns:
        쿼리 결과

    Raises:
        DatabaseConnectionError: DB 연결 실패 시
    """
    # 먼저 DB 가용성 체크
    if not is_db_available():
        raise DatabaseConnectionError(f"{operation_name} 실패: 데이터베이스에 연결할 수 없습니다")

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(query, params or {})
            return result
    except OperationalError as e:
        error_msg = str(e)
        if "Can't connect" in error_msg or "Connection refused" in error_msg:
            raise DatabaseConnectionError(f"{operation_name} 실패", original_error=e)
        raise

# =============================================================================
# 수집 이력 관리 (API 호출 중복 방지)
# =============================================================================
def check_price_log(sigungu_code: str, deal_ymd: str, data_type: str) -> bool:
    """
    해당 지역/년월/타입의 데이터가 오늘 수집되었는지 확인

    Args:
        sigungu_code: 시군구 코드
        deal_ymd: 거래년월 (YYYYMM)
        data_type: 'TRADE' 또는 'RENT'

    Returns:
        오늘 수집 여부

    Raises:
        DatabaseConnectionError: DB 연결 실패 시
    """
    engine = get_engine()

    query = text("""
        SELECT 1 FROM api_price_log 
        WHERE sigungu_code = :sgg 
          AND deal_ymd = :ymd 
          AND data_type = :dtype
          AND DATE(collected_at) = CURDATE()
        LIMIT 1
    """)

    try:
        result = _execute_query_safe(
            query,
            {"sgg": sigungu_code, "ymd": deal_ymd, "dtype": data_type},
            "수집 이력 확인"
        )
        return result.fetchone() is not None
    except DatabaseConnectionError:
        raise
    except Exception as e:
        logger.error(f"수집 이력 확인 실패: {e}")
        return False


def update_price_log(sigungu_code: str, deal_ymd: str, data_type: str) -> None:
    if not is_db_available():
        logger.warning("DB 연결 불가로 수집 이력 저장 스킵")
        return
    """수집 완료 이력 기록"""
    engine = get_engine()

    # MySQL용 UPSERT
    query = text("""
        INSERT INTO api_price_log (sigungu_code, deal_ymd, data_type, collected_at)
        VALUES (:sgg, :ymd, :dtype, NOW())
        ON DUPLICATE KEY UPDATE collected_at = NOW()
    """)

    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(query, {
                    "sgg": sigungu_code,
                    "ymd": deal_ymd,
                    "dtype": data_type
                })
    except Exception as e:
        logger.error(f"수집 이력 저장 실패: {e}")


# =============================================================================
# 실거래가 API 수집
# =============================================================================
def fetch_real_price_from_api(sigungu_code: str, bjdong_code: str) -> bool:
    """
    국토부 API에서 실거래가 수집 (최근 10개월)

    Args:
        sigungu_code: 시군구 코드
        bjdong_code: 법정동 코드

    Returns:
        수집 성공 여부

    Raises:
        DatabaseConnectionError: DB 연결 실패 시 (호출자가 처리)
    """
    # DB 연결 확인 - 불가 시 즉시 종료
    if not is_db_available():
        logger.warning("DB 연결 불가로 실거래가 API 수집 스킵")
        raise DatabaseConnectionError("데이터베이스에 연결할 수 없어 실거래가를 수집할 수 없습니다")

    # 외부 API 모듈 import
    try:
        from scripts.data_collecting.fetch_trade_data import (
            fetch_trade_data_and_save,
            get_bjdong_code_map
        )
        from scripts.data_collecting.fetch_rent_data import fetch_rent_data_and_save
    except ImportError as e:
        logger.error(f"API 모듈 import 실패: {e}")
        return False

    logger.info(f"실거래가 데이터 최신화 점검 (시군구: {sigungu_code})")

    code_map = get_bjdong_code_map()

    # 최근 10개월 데이터 수집
    for i in range(10):
        target_date = datetime.now() - timedelta(days=30 * i)
        deal_ymd = target_date.strftime('%Y%m')

        # 매매 데이터
        if not check_price_log(sigungu_code, deal_ymd, 'TRADE'):
            try:
                logger.info(f"매매 데이터 수집: {deal_ymd}")
                fetch_trade_data_and_save(sigungu_code, deal_ymd, code_map)
                update_price_log(sigungu_code, deal_ymd, 'TRADE')
            except Exception as e:
                logger.error(f"매매 데이터 수집 오류: {e}")

        # 전월세 데이터
        if not check_price_log(sigungu_code, deal_ymd, 'RENT'):
            try:
                logger.info(f"전월세 데이터 수집: {deal_ymd}")
                fetch_rent_data_and_save(sigungu_code, deal_ymd, code_map)
                update_price_log(sigungu_code, deal_ymd, 'RENT')
            except Exception as e:
                logger.error(f"전월세 데이터 수집 오류: {e}")

    logger.info("실거래가 점검 완료")
    return True


# =============================================================================
# 실거래가 DB 조회
# =============================================================================
def get_trade_price(
        pnu: str,
        area_size: Optional[float] = None
) -> Tuple[float, str]:
    """
    실거래가(매매) 조회

    Args:
        pnu: PNU 문자열
        area_size: 전용면적 (옵션)

    Returns:
        Tuple[가격(만원), 출처]

    Raises:
        DatabaseConnectionError: DB 연결 실패 시
    """
    if not pnu:
        return 0, "Unknown"

    parsed = parse_pnu(pnu)
    if not parsed:
        return 0, "Unknown"

    # DB 연결 확인
    if not is_db_available():
        raise DatabaseConnectionError("실거래가 조회 실패: 데이터베이스에 연결할 수 없습니다")

    engine = get_engine()

    # 면적 조건 추가 여부
    area_condition = ""
    params = {
        "sigungu": parsed["sigungu_code"],
        "bjdong": parsed["bjdong_code"],
        "bonbun": parsed["bonbun"],
        "bubun": parsed["bubun"]
    }

    if area_size:
        area_condition = "AND CAST(전용면적 AS DECIMAL(10,2)) BETWEEN :area - 3 AND :area + 3"
        params["area"] = area_size

    query = text(f"""
        SELECT 거래금액 as price 
        FROM raw_trade 
        WHERE 시군구 = :sigungu 
          AND 법정동 = :bjdong 
          AND 본번 = :bonbun 
          AND 부번 = :bubun
          {area_condition}
        ORDER BY 계약일 DESC 
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            row = result.mappings().fetchone()

            if row:
                price_str = str(row['price']).replace(',', '')
                price = float(price_str)
                logger.info(f"실거래가 발견: {price:,.0f}만원")
                return price, "DB_Trade"
    except OperationalError as e:
        error_msg = str(e)
        if "Can't connect" in error_msg or "Connection refused" in error_msg:
            raise DatabaseConnectionError("실거래가 조회 실패", original_error=e)
        logger.error(f"실거래가 조회 실패: {e}")
    except Exception as e:
        logger.error(f"실거래가 조회 실패: {e}")

    return 0, "Unknown"


def get_rent_price(
        pnu: str,
        area_size: Optional[float] = None
) -> Tuple[float, str]:
    """
    전세가 조회

    Args:
        pnu: PNU 문자열
        area_size: 전용면적 (옵션)

    Returns:
        Tuple[보증금(만원), 출처]

    Raises:
        DatabaseConnectionError: DB 연결 실패 시
    """
    if not pnu:
        return 0, "Unknown"

    parsed = parse_pnu(pnu)
    if not parsed:
        return 0, "Unknown"

    # DB 연결 확인
    if not is_db_available():
        raise DatabaseConnectionError("전세가 조회 실패: 데이터베이스에 연결할 수 없습니다")

    engine = get_engine()

    area_condition = ""
    params = {
        "sigungu": parsed["sigungu_code"],
        "bjdong": parsed["bjdong_code"],
        "bonbun": parsed["bonbun"].lstrip('0') or '0',
        "bubun": parsed["bubun"].lstrip('0') or '0'
    }

    if area_size:
        area_condition = "AND CAST(전용면적 AS DECIMAL(10,2)) BETWEEN :area - 3 AND :area + 3"
        params["area"] = area_size

    query = text(f"""
        SELECT 보증금 as deposit
        FROM raw_rent 
        WHERE 시군구 = :sigungu 
          AND 법정동 = :bjdong 
          AND 본번 = :bonbun 
          AND 부번 = :bubun
          AND (월세 = '0' OR 월세 = '' OR 월세 IS NULL)
          {area_condition}
        ORDER BY 계약일 DESC 
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            row = result.mappings().fetchone()

            if row:
                deposit_str = str(row['deposit']).replace(',', '')
                deposit = float(deposit_str)
                logger.info(f"전세가 발견: {deposit:,.0f}만원")
                return deposit, "DB_Rent"
    except OperationalError as e:
        error_msg = str(e)
        if "Can't connect" in error_msg or "Connection refused" in error_msg:
            raise DatabaseConnectionError("전세가 조회 실패", original_error=e)
        logger.error(f"전세가 조회 실패: {e}")
    except Exception as e:
        logger.error(f"전세가 조회 실패: {e}")

    return 0, "Unknown"


# =============================================================================
# 공시지가 조회
# =============================================================================
def get_public_price(
        pnu: str,
        area_size: Optional[float] = None
) -> float:
    """
    공시지가 조회

    Args:
        pnu: PNU 문자열
        area_size: 전용면적 (면적 매칭용)

    Returns:
        공시가격 (원)

    Raises:
        DatabaseConnectionError: DB 연결 실패 시
    """
    if not pnu:
        return 0

    pnu_raw = pnu_to_raw_format(pnu)

    # DB 연결 확인
    if not is_db_available():
        raise DatabaseConnectionError("공시지가 조회 실패: 데이터베이스에 연결할 수 없습니다")

    engine = get_engine()

    query = text("""
        SELECT price, exclusive_area 
        FROM official_price_raw
        WHERE pnu = :pnu 
        ORDER BY base_year DESC
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"pnu": pnu_raw})
            rows = result.mappings().fetchall()

            if not rows:
                logger.info(f"공시지가 없음 (PNU: {pnu_raw})")
                return 0

            # 단일 결과
            if len(rows) == 1:
                price = float(rows[0]['price'])
                logger.info(f"공시지가 발견: {price:,.0f}원")
                return price

            # 복수 결과 -> 면적 매칭
            if area_size:
                for row in rows:
                    row_area = float(row['exclusive_area'] or 0)
                    if abs(row_area - area_size) < 3.3:  # 1평 오차 허용
                        price = float(row['price'])
                        logger.info(f"공시지가 발견 (면적 매칭): {price:,.0f}원")
                        return price

            # 매칭 실패 시 첫 번째 결과
            price = float(rows[0]['price'])
            logger.info(f"공시지가 발견 (첫 번째): {price:,.0f}원")
            return price
    except OperationalError as e:
        error_msg = str(e)
        if "Can't connect" in error_msg or "Connection refused" in error_msg:
            raise DatabaseConnectionError("공시지가 조회 실패", original_error=e)
        logger.error(f"공시지가 조회 실패: {e}")
    except Exception as e:
        logger.error(f"공시지가 조회 실패: {e}")

    return 0


# =============================================================================
# 시세 추정 통합 함수
# =============================================================================
def estimate_market_price(
        pnu: str,
        area_size: Optional[float] = None,
        fetch_if_missing: bool = True
) -> Tuple[float, str]:
    """
    시세 추정 (실거래가 -> 공시지가 순으로 시도)

    Args:
        pnu: PNU 문자열
        area_size: 전용면적
        fetch_if_missing: 데이터 없을 시 API 수집 여부

    Returns:
        Tuple[시세(만원), 출처]

    Raises:
        DatabaseConnectionError: DB 연결 실패 시
    """
    parsed = parse_pnu(pnu)

    # 1. 실거래가 조회
    trade_price, source = get_trade_price(pnu, area_size)

    if trade_price > 0:
        return trade_price, source

    # 2. 실거래가 없으면 API 수집 시도
    if fetch_if_missing and parsed and is_db_available():
        try:
            fetch_real_price_from_api(parsed["sigungu_code"], parsed["bjdong_code"])

            # 재조회
            trade_price, source = get_trade_price(pnu, area_size)
            if trade_price > 0:
                return trade_price, source
        except DatabaseConnectionError:
            # DB 연결 오류는 상위로 전파
            raise
        except Exception as e:
            logger.warning(f"API 수집 실패: {e}")

    # 3. 공시지가 기반 추정
    public_price = get_public_price(pnu, area_size)

    if public_price > 0:
        # 공시가의 126%를 시세로 추정 (HUG 기준)
        estimated = (public_price / 10000) * 1.26
        logger.info(f"공시지가 기반 시세 추정: {estimated:,.0f}만원")
        return estimated, "Public_Price_Estimate"

    return 0, "Unknown"


def calculate_hug_eligibility(
        public_price: float,
        deposit_manwon: float
) -> Tuple[bool, float, str]:
    """
    HUG 보증보험 가입 가능 여부 판단

    Args:
        public_price: 공시가격 (원)
        deposit_manwon: 보증금 (만원)

    Returns:
        Tuple[가입가능여부, HUG한도(만원), 메시지]
    """
    if public_price <= 0:
        return False, 0, "공시가 없음 (판단 불가)"

    # HUG 한도 = 공시가 × 126%
    hug_limit_won = public_price * 1.26
    hug_limit_manwon = hug_limit_won / 10000

    if deposit_manwon <= hug_limit_manwon:
        return True, hug_limit_manwon, "가입 가능 (안전 ✅)"
    else:
        return False, hug_limit_manwon, "가입 불가 (위험 ❌)"