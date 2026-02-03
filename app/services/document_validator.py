"""
문서 매칭 검증 서비스

담당 기능:
- 등기부등본과 건축물대장의 동일 물건 여부 검증
- 주소, 고유번호, 소유자 정보 크로스체크
"""
import re
import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """검증 결과 데이터 클래스"""
    is_valid: bool
    confidence: float  # 0.0 ~ 1.0
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class DocumentValidator:
    """등기부등본과 건축물대장 매칭 검증기"""

    # 검증 임계값
    ADDRESS_SIMILARITY_THRESHOLD = 0.7  # 주소 유사도 최소 70%
    CONFIDENCE_THRESHOLD = 0.6  # 전체 신뢰도 최소 60%

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.match_scores: Dict[str, float] = {}

    def validate(
            self,
            ledger_data: Dict[str, Any],
            registry_data: Dict[str, Any]
    ) -> ValidationResult:
        """
        두 문서의 매칭 여부를 종합 검증

        Args:
            ledger_data: 건축물대장 OCR 결과
            registry_data: 등기부등본 OCR 결과

        Returns:
            ValidationResult: 검증 결과
        """
        self.errors = []
        self.warnings = []
        self.match_scores = {}

        # 1. 데이터 존재 여부 확인
        if not ledger_data or not registry_data:
            return self._create_result(
                is_valid=False,
                confidence=0.0,
                error_msg="건축물대장 또는 등기부등본 데이터가 없습니다"
            )

        # 2. 필수 필드 추출
        ledger_info = self._extract_ledger_info(ledger_data)
        registry_info = self._extract_registry_info(registry_data)

        # 3. 각 항목별 매칭 검증
        self._validate_address(ledger_info, registry_info)
        self._validate_unique_number(ledger_info, registry_info)
        self._validate_owner(ledger_info, registry_info)
        self._validate_area(ledger_info, registry_info)

        # 4. 종합 신뢰도 계산
        confidence = self._calculate_confidence()

        # 5. 최종 판정
        is_valid = confidence >= self.CONFIDENCE_THRESHOLD and len(self.errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            errors=self.errors,
            warnings=self.warnings,
            details={
                "match_scores": self.match_scores,
                "ledger_info": ledger_info,
                "registry_info": registry_info
            }
        )

    def _extract_ledger_info(self, ledger: Dict) -> Dict[str, Any]:
        """건축물대장에서 비교용 정보 추출"""
        location = ledger.get('location', {})
        document_info = ledger.get('document_info', {})
        building_status = ledger.get('building_status', {})
        safety_check = ledger.get('safety_check', {})

        # 주소 추출 (도로명 또는 지번)
        address = location.get('address', '') or ''
        detail_address = location.get('detail_address', '') or ''
        full_address = f"{address} {detail_address}".strip()

        # 면적 추출
        area = None
        area_raw = building_status.get('area')
        if area_raw:
            try:
                nums = re.findall(r'[\d.]+', str(area_raw))
                if nums:
                    area = float(nums[0])
            except:
                pass

        return {
            "address": full_address,
            "address_normalized": self._normalize_address(full_address),
            "unique_number": document_info.get('unique_number', ''),
            "owner": safety_check.get('owner_name', ''),
            "area": area
        }

    def _extract_registry_info(self, registry: Dict) -> Dict[str, Any]:
        """등기부등본에서 비교용 정보 추출"""
        basic_info = registry.get('basic_info', {})

        address = basic_info.get('address', '') or ''

        return {
            "address": address,
            "address_normalized": self._normalize_address(address),
            "unique_number": "",  # 등기부등본에는 고유번호가 다른 형식
            "owner": basic_info.get('owner', ''),
            "area": None  # 등기부등본에서 면적 추출 필요시 추가
        }

    def _normalize_address(self, address: str) -> str:
        """
        주소 정규화 (비교를 위한 전처리)

        - 공백 통일
        - 불필요한 문자 제거
        - 숫자 형식 통일
        """
        if not address:
            return ""

        addr = str(address)

        # 1. 공백 정규화
        addr = re.sub(r'\s+', ' ', addr).strip()

        # 2. 괄호 안 내용 제거 (상세 정보)
        addr = re.sub(r'\([^)]*\)', '', addr)

        # 3. 불필요한 접미사 제거
        addr = re.sub(r'(아파트|빌라|오피스텔|주택|빌딩|타워|파크|힐스|캐슬)$', '', addr)

        # 4. 층/호 정보 분리
        addr = re.sub(r'\s*(제?\d+층)\s*', ' ', addr)

        # 5. 번지 형식 통일 (123-45 → 123-45)
        addr = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', addr)

        # 6. "번지" 제거
        addr = re.sub(r'번지', '', addr)

        return addr.strip()

    def _calculate_address_similarity(self, addr1: str, addr2: str) -> float:
        """
        두 주소의 유사도 계산 (0.0 ~ 1.0)
        """
        if not addr1 or not addr2:
            return 0.0

        # 기본 유사도
        base_similarity = SequenceMatcher(None, addr1, addr2).ratio()

        # 핵심 요소 매칭 보너스
        bonus = 0.0

        # 시/도 매칭
        city_pattern = r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)'
        city1 = re.search(city_pattern, addr1)
        city2 = re.search(city_pattern, addr2)
        if city1 and city2 and city1.group() == city2.group():
            bonus += 0.1

        # 구/군 매칭
        district_pattern = r'([가-힣]+[구군시])\s'
        dist1 = re.search(district_pattern, addr1)
        dist2 = re.search(district_pattern, addr2)
        if dist1 and dist2 and dist1.group(1) == dist2.group(1):
            bonus += 0.1

        # 동/읍/면 매칭
        dong_pattern = r'([가-힣]+[동읍면리])\s'
        dong1 = re.search(dong_pattern, addr1)
        dong2 = re.search(dong_pattern, addr2)
        if dong1 and dong2 and dong1.group(1) == dong2.group(1):
            bonus += 0.1

        # 번지 매칭
        lot_pattern = r'(\d+(?:-\d+)?)'
        lot1 = re.findall(lot_pattern, addr1)
        lot2 = re.findall(lot_pattern, addr2)
        if lot1 and lot2 and set(lot1) & set(lot2):
            bonus += 0.15

        return min(1.0, base_similarity + bonus)

    def _validate_address(
            self,
            ledger_info: Dict,
            registry_info: Dict
    ) -> None:
        """주소 매칭 검증"""
        ledger_addr = ledger_info.get('address_normalized', '')
        registry_addr = registry_info.get('address_normalized', '')

        if not ledger_addr and not registry_addr:
            self.warnings.append("양쪽 문서 모두에서 주소를 추출할 수 없습니다")
            self.match_scores['address'] = 0.5  # 불확실
            return

        if not ledger_addr:
            self.warnings.append("건축물대장에서 주소를 추출할 수 없습니다")
            self.match_scores['address'] = 0.5
            return

        if not registry_addr:
            self.warnings.append("등기부등본에서 주소를 추출할 수 없습니다")
            self.match_scores['address'] = 0.5
            return

        similarity = self._calculate_address_similarity(ledger_addr, registry_addr)
        self.match_scores['address'] = similarity

        if similarity < self.ADDRESS_SIMILARITY_THRESHOLD:
            self.errors.append(
                f"주소가 일치하지 않습니다 (유사도: {similarity:.1%})\n"
                f"  - 건축물대장: {ledger_info.get('address', '')}\n"
                f"  - 등기부등본: {registry_info.get('address', '')}"
            )
        elif similarity < 0.9:
            self.warnings.append(
                f"주소가 완전히 일치하지 않습니다 (유사도: {similarity:.1%})"
            )

    def _validate_unique_number(
            self,
            ledger_info: Dict,
            registry_info: Dict
    ) -> None:
        """고유번호 검증 (건축물대장에만 있음, 참고용)"""
        unique_number = ledger_info.get('unique_number', '')

        if unique_number:
            # 고유번호 형식 검증 (예: 1234510200-1-12345678)
            pattern = r'^\d{10}-\d-\d{8}$'
            if not re.match(pattern, unique_number.replace(' ', '')):
                self.warnings.append(
                    f"건축물대장 고유번호 형식이 비정상입니다: {unique_number}"
                )
            else:
                self.match_scores['unique_number'] = 1.0
        else:
            self.match_scores['unique_number'] = 0.5  # 없으면 중립

    def _validate_owner(
            self,
            ledger_info: Dict,
            registry_info: Dict
    ) -> None:
        """소유자 매칭 검증"""
        ledger_owner = self._normalize_name(ledger_info.get('owner', ''))
        registry_owner = self._normalize_name(registry_info.get('owner', ''))

        if not ledger_owner and not registry_owner:
            self.match_scores['owner'] = 0.5
            return

        if not ledger_owner or not registry_owner:
            self.warnings.append("소유자 정보를 비교할 수 없습니다")
            self.match_scores['owner'] = 0.5
            return

        # 이름 비교 (신탁 등 특수 케이스 고려)
        if ledger_owner == registry_owner:
            self.match_scores['owner'] = 1.0
        elif '신탁' in ledger_owner or '신탁' in registry_owner:
            # 신탁의 경우 이름이 다를 수 있음
            self.warnings.append("신탁 소유로 소유자명이 다를 수 있습니다")
            self.match_scores['owner'] = 0.7
        else:
            # 부분 일치 확인
            similarity = SequenceMatcher(None, ledger_owner, registry_owner).ratio()
            self.match_scores['owner'] = similarity

            if similarity < 0.5:
                self.errors.append(
                    f"소유자가 일치하지 않습니다\n"
                    f"  - 건축물대장: {ledger_info.get('owner', '(없음)')}\n"
                    f"  - 등기부등본: {registry_info.get('owner', '(없음)')}"
                )

    def _normalize_name(self, name: str) -> str:
        """이름 정규화"""
        if not name:
            return ""

        name = str(name).strip()
        # 괄호 내용 제거
        name = re.sub(r'\([^)]*\)', '', name)
        # 공백 제거
        name = re.sub(r'\s+', '', name)

        return name

    def _validate_area(
            self,
            ledger_info: Dict,
            registry_info: Dict
    ) -> None:
        """면적 검증 (건축물대장 기준)"""
        ledger_area = ledger_info.get('area')

        if ledger_area and ledger_area > 0:
            self.match_scores['area'] = 1.0

            # 면적 이상치 체크
            if ledger_area < 10:
                self.warnings.append(f"전용면적이 매우 작습니다: {ledger_area}㎡")
            elif ledger_area > 300:
                self.warnings.append(f"전용면적이 매우 큽니다: {ledger_area}㎡")
        else:
            self.match_scores['area'] = 0.5

    def _calculate_confidence(self) -> float:
        """
        종합 신뢰도 계산

        가중치:
        - 주소: 50%
        - 소유자: 30%
        - 면적: 10%
        - 고유번호: 10%
        """
        weights = {
            'address': 0.5,
            'owner': 0.3,
            'area': 0.1,
            'unique_number': 0.1
        }

        total_score = 0.0
        total_weight = 0.0

        for key, weight in weights.items():
            if key in self.match_scores:
                total_score += self.match_scores[key] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def _create_result(
            self,
            is_valid: bool,
            confidence: float,
            error_msg: str = None
    ) -> ValidationResult:
        """결과 객체 생성 헬퍼"""
        errors = [error_msg] if error_msg else []

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            errors=errors,
            warnings=[],
            details={}
        )


def validate_document_match(
        ledger_data: Dict[str, Any],
        registry_data: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    문서 매칭 검증 (외부 호출용 간편 함수)

    Args:
        ledger_data: 건축물대장 OCR 결과
        registry_data: 등기부등본 OCR 결과

    Returns:
        Tuple[bool, str, Dict]:
            - is_valid: 매칭 여부
            - message: 결과 메시지
            - details: 상세 정보
    """
    validator = DocumentValidator()
    result = validator.validate(ledger_data, registry_data)

    # 메시지 생성
    if result.is_valid:
        message = f"문서 매칭 확인 완료 (신뢰도: {result.confidence:.1%})"
        if result.warnings:
            message += f" - 주의사항 {len(result.warnings)}건"
    else:
        if result.errors:
            message = result.errors[0]
        else:
            message = f"문서 매칭 실패 (신뢰도: {result.confidence:.1%})"

    return result.is_valid, message, {
        "confidence": result.confidence,
        "errors": result.errors,
        "warnings": result.warnings,
        "match_scores": result.details.get("match_scores", {})
    }


# =============================================================================
# API 응답 생성 함수
# =============================================================================
def create_mismatch_error_response(
        validation_result: ValidationResult,
        timestamp: str = None
) -> Dict[str, Any]:
    """
    문서 불일치 시 API 에러 응답 생성

    Returns:
        FastAPI 응답용 딕셔너리
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.now().isoformat()

    return {
        "meta": {
            "code": 422,
            "message": "문서 불일치 오류",
            "timestamp": timestamp
        },
        "errors": [
            {
                "field": "documents",
                "message": "업로드된 건축물대장과 등기부등본이 동일한 물건이 아닙니다",
                "details": {
                    "confidence": round(validation_result.confidence * 100, 1),
                    "issues": validation_result.errors,
                    "warnings": validation_result.warnings,
                    "match_scores": {
                        k: round(v * 100, 1)
                        for k, v in validation_result.details.get("match_scores", {}).items()
                    }
                }
            }
        ],
        "suggestions": [
            "건축물대장과 등기부등본이 같은 주소의 문서인지 확인해주세요",
            "호수(동/호)가 정확히 일치하는지 확인해주세요",
            "최신 문서를 사용하고 있는지 확인해주세요"
        ]
    }


# =============================================================================
# 테스트 코드
# =============================================================================
if __name__ == "__main__":
    # 테스트 케이스 1: 정상 매칭
    ledger_match = {
        "document_info": {"unique_number": "2823710200-3-03540011"},
        "location": {
            "address": "인천광역시 부평구 삼산동 167-15",
            "detail_address": "101호"
        },
        "building_status": {"area": "59.84"},
        "safety_check": {"owner_name": "홍길동"}
    }

    registry_match = {
        "basic_info": {
            "address": "인천광역시 부평구 삼산동 167-15 101호",
            "owner": "홍길동"
        }
    }

    print("=" * 60)
    print("테스트 1: 정상 매칭 케이스")
    print("=" * 60)
    is_valid, msg, details = validate_document_match(ledger_match, registry_match)
    print(f"결과: {'✅ 통과' if is_valid else '❌ 실패'}")
    print(f"메시지: {msg}")
    print(f"신뢰도: {details['confidence']:.1%}")
    print()

    # 테스트 케이스 2: 주소 불일치
    ledger_mismatch = {
        "document_info": {"unique_number": "2823710200-3-03540011"},
        "location": {
            "address": "인천광역시 부평구 삼산동 167-15",
            "detail_address": "101호"
        },
        "building_status": {"area": "59.84"},
        "safety_check": {"owner_name": "홍길동"}
    }

    registry_mismatch = {
        "basic_info": {
            "address": "서울특별시 강남구 역삼동 123-45",  # 완전히 다른 주소
            "owner": "김철수"  # 다른 소유자
        }
    }

    print("=" * 60)
    print("테스트 2: 주소/소유자 불일치 케이스")
    print("=" * 60)
    is_valid, msg, details = validate_document_match(ledger_mismatch, registry_mismatch)
    print(f"결과: {'✅ 통과' if is_valid else '❌ 실패'}")
    print(f"메시지: {msg}")
    print(f"신뢰도: {details['confidence']:.1%}")
    print(f"에러: {details['errors']}")
    print()

    # 테스트 케이스 3: 부분 일치 (호수만 다름)
    ledger_partial = {
        "location": {
            "address": "인천광역시 부평구 삼산동 167-15",
            "detail_address": "101호"
        },
        "safety_check": {"owner_name": "홍길동"}
    }

    registry_partial = {
        "basic_info": {
            "address": "인천광역시 부평구 삼산동 167-15 201호",  # 호수만 다름
            "owner": "홍길동"
        }
    }

    print("=" * 60)
    print("테스트 3: 부분 일치 (호수 다름)")
    print("=" * 60)
    is_valid, msg, details = validate_document_match(ledger_partial, registry_partial)
    print(f"결과: {'✅ 통과' if is_valid else '❌ 실패'}")
    print(f"메시지: {msg}")
    print(f"신뢰도: {details['confidence']:.1%}")
    print(f"매칭 점수: {details['match_scores']}")