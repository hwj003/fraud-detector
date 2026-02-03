"""
주소 정규화 및 변환 서비스

담당 기능:
- 시도명 정규화 (서울 -> 서울특별시)
- 주소를 PNU(고유번호)로 변환
- 법정동 코드 관리
"""
import os
import re
import logging
from typing import Optional, Tuple, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# =============================================================================
# 시도명 매핑
# =============================================================================
SIDO_MAP = {
    "서울": "서울특별시",
    "서울시": "서울특별시",
    "인천": "인천광역시",
    "인천시": "인천광역시",
    "경기": "경기도",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도"
}

# =============================================================================
# 법정동 코드 로더 (Lazy Loading + Caching)
# =============================================================================
_bjd_map: Optional[Dict[str, str]] = None


def get_bjd_map() -> Dict[str, str]:
    """
    법정동 코드 매핑 딕셔너리 반환 (싱글톤)

    Returns:
        Dict[str, str]: {법정동명: 법정동코드} 매핑
    """
    global _bjd_map

    if _bjd_map is not None:
        return _bjd_map

    csv_path = os.path.join(PROJECT_ROOT, 'data', '국토교통부_법정동코드_20250805.csv')

    if not os.path.exists(csv_path):
        logger.warning(f"법정동 코드 파일이 없습니다: {csv_path}")
        _bjd_map = {}
        return _bjd_map

    try:
        df = pd.read_csv(csv_path, sep=',', encoding='cp949', dtype=str)
        _bjd_map = dict(zip(df['법정동명'], df['법정동코드']))
        logger.info(f"법정동 코드 로드 완료: {len(_bjd_map)}건")
    except Exception as e:
        logger.error(f"법정동 코드 로드 실패: {e}")
        _bjd_map = {}

    return _bjd_map


# =============================================================================
# 주소 정규화 함수들
# =============================================================================
def normalize_address(address: Optional[str]) -> Optional[str]:
    """
    주소 문자열 정규화

    - 시도 약칭을 정식 명칭으로 변환 (서울 -> 서울특별시)
    - 불필요한 공백 제거

    Args:
        address: 원본 주소 문자열

    Returns:
        정규화된 주소 문자열
    """
    if not address or not isinstance(address, str):
        return address

    address = address.strip()
    tokens = address.split()

    if not tokens:
        return address

    # 첫 번째 토큰(시도명) 정규화
    if tokens[0] in SIDO_MAP:
        tokens[0] = SIDO_MAP[tokens[0]]

    return " ".join(tokens)


def extract_address_components(address: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    주소에서 구성요소 추출

    Args:
        address: 전체 주소 (예: "서울특별시 강남구 역삼동 123-45")

    Returns:
        Tuple[지역부분, 본번, 부번]
    """
    # 번지 패턴 매칭 (123 또는 123-45)
    match = re.search(r'(.+)\s+(\d+)(?:-(\d+))?$', address.strip())

    if not match:
        return None, None, None

    region_part = match.group(1).strip()
    main_no = match.group(2)
    sub_no = match.group(3) or "0"

    return region_part, main_no, sub_no


# =============================================================================
# PNU 변환 함수들
# =============================================================================
def convert_address_to_pnu(address: str) -> Tuple[Optional[str], str]:
    """
    주소를 PNU(Parcel Number Unit)로 변환

    PNU 형식: {법정동코드(10자리)}-{지목코드(1자리)}-{본번(4자리)}{부번(4자리)}
    예: 1111010100-3-00010002

    Args:
        address: 지번 주소 (예: "서울특별시 종로구 청운동 1-2")

    Returns:
        Tuple[PNU 문자열 또는 None, 결과 메시지]
    """
    bjd_map = get_bjd_map()

    if not bjd_map:
        return None, "법정동 코드 데이터가 없습니다"

    # 주소 구성요소 추출
    region_part, main_no, sub_no = extract_address_components(address)

    if not region_part or not main_no:
        return None, "주소 형식 오류 (번지를 찾을 수 없습니다)"

    # 법정동 코드 조회
    if region_part not in bjd_map:
        return None, f"법정동 코드를 찾을 수 없습니다: {region_part}"

    bjd_code = bjd_map[region_part]

    # PNU 생성 (지목코드 3 = 대지)
    pnu = f"{bjd_code}-3-{int(main_no):04d}{int(sub_no):04d}"

    return pnu, "성공"


def parse_pnu(pnu: str) -> Optional[Dict[str, str]]:
    """
    PNU 문자열을 구성요소로 파싱

    Args:
        pnu: PNU 문자열 (예: "1111010100-3-00010002")

    Returns:
        구성요소 딕셔너리 또는 None
    """
    if not pnu:
        return None

    parts = pnu.split('-')

    if len(parts) < 3:
        return None

    sgg_bjd = parts[0]  # 시군구코드(5) + 법정동코드(5)
    land_type = parts[1]  # 지목코드
    bunji = parts[2]  # 본번(4) + 부번(4)

    return {
        "sigungu_code": sgg_bjd[:5],
        "bjdong_code": sgg_bjd[5:10],
        "land_type": land_type,
        "bonbun": bunji[:4],
        "bubun": bunji[4:8] if len(bunji) >= 8 else "0000",
        "full_pnu": pnu
    }


def pnu_to_raw_format(pnu: str) -> str:
    """
    PNU를 DB 조회용 포맷으로 변환 (하이픈 제거)

    Args:
        pnu: "1111010100-3-00010002"

    Returns:
        "1111010100000010002" (19자리)
    """
    parts = pnu.split('-')

    if len(parts) >= 3:
        # 중간 부분(지목코드)을 '0'으로 통일
        return parts[0] + "0" + parts[2]

    return pnu.replace('-', '')


def create_address_key(pnu: str) -> str:
    """
    PNU를 address_key 형식으로 변환 (결과 저장용)

    Args:
        pnu: "2823710100-3-00650124"

    Returns:
        "28237-10100-0065-0124"
    """
    if not pnu or '-' not in pnu:
        return "UNKNOWN"

    parts = pnu.split('-')

    if len(parts) < 3:
        return "UNKNOWN"

    sgg_bjd = parts[0]
    bon_bu = parts[2]

    return f"{sgg_bjd[:5]}-{sgg_bjd[5:]}-{bon_bu[:4]}-{bon_bu[4:]}"