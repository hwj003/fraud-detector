from pydantic import BaseModel, Field
from typing import Literal, List, Any, Dict, Optional
from datetime import datetime
from enum import Enum

# --- Envelope 패턴 ---
class Meta(BaseModel):
    code: int
    message: str

class ResponseEnvelope(BaseModel):
    meta: Meta
    data: Any = None

# 지도용 요약 정보 (History 없음)
class RegionSummary(BaseModel):
    region_code: str
    region_name: str
    coordinates: dict # {"lat": ..., "lng": ...}

    # 요약 정보만 포함 (가벼움)
    latest_ratio: float
    risk_level: str
    total_tx_count: int

# 상세 히스토리 정보
class MonthlyData(BaseModel):
    month: str
    ratio: float

class RegionHistory(BaseModel):
    region_code: str
    region_name: str
    history: List[MonthlyData] # 여기에만 무거운 배열이 들어감


class RiskLevel(str, Enum):
    """위험도 등급"""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    RISKY = "RISKY"


class RiskSeverity(str, Enum):
    """위험 요인 심각도"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskFactorType(str, Enum):
    """위험 요인 유형"""
    OWNERSHIP_PERIOD = "OWNERSHIP_PERIOD"
    HUG_INELIGIBLE = "HUG_INELIGIBLE"
    HIGH_LTV = "HIGH_LTV"
    SENIOR_DEBT = "SENIOR_DEBT"
    ILLEGAL_BUILDING = "ILLEGAL_BUILDING"
    TRUST_PROPERTY = "TRUST_PROPERTY"
    OLD_BUILDING = "OLD_BUILDING"
    NONE = "NONE"


class RiskFactor(BaseModel):
    """구조화된 위험 요인"""
    type: RiskFactorType = Field(..., description="위험 유형 코드")
    severity: RiskSeverity = Field(..., description="심각도")
    message: str = Field(..., description="위험 요인 설명")


class HugResult(BaseModel):
    """HUG 보증보험 진단 결과"""
    is_eligible: bool = Field(..., description="가입 가능 여부")
    safe_limit: int = Field(..., description="가입 한도액 (원)")
    coverage_ratio: Optional[float] = Field(None, description="보증 커버리지 (%)")
    reason: Optional[str] = Field(None, description="불가 사유 (불가능한 경우)")
    message: str = Field(..., description="요약 메시지")


class AnalysisDetails(BaseModel):
    """상세 분석 지표"""
    jeonse_ratio: float = Field(..., description="전세가율 (%)")
    senior_debt: int = Field(..., description="선순위 채권 (원)")
    is_illegal_building: bool = Field(..., description="위반 건축물 여부")
    is_trust: bool = Field(..., description="신탁 여부")
    building_age: float = Field(..., description="건물 연식 (년)")
    ownership_duration_months: Optional[int] = Field(None, description="소유 기간 (개월)")


class ResponseMeta(BaseModel):
    """응답 메타 정보"""
    code: int = Field(200, description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    timestamp: datetime = Field(default_factory=datetime.now, description="분석 시점")


class PredictionData(BaseModel):
    """예측 결과 데이터"""
    address: str = Field(..., description="분석 주소")
    deposit: int = Field(..., description="보증금 (원)")
    market_price: int = Field(..., description="시세 (원)")
    price_source: str = Field(..., description="시세 출처")

    risk_score: float = Field(..., ge=0, le=100, description="위험도 점수 (0~100)")
    risk_level: RiskLevel = Field(..., description="위험도 등급")
    major_risk_factors: List[RiskFactor] = Field(..., description="주요 위험 요인")

    hug_result: HugResult = Field(..., description="HUG 보증보험 진단")
    details: AnalysisDetails = Field(..., description="상세 분석 지표")
    recommendations: List[str] = Field(default_factory=list, description="권장 조치사항")


class PredictionResponseV2(BaseModel):
    """개선된 예측 응답"""
    meta: ResponseMeta
    data: PredictionData
    debug_info: Optional[Dict[str, Any]] = Field(None, description="디버그 정보 (개발용)")


class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    field: str = Field(..., description="에러 발생 필드")
    message: str = Field(..., description="에러 메시지")


class ErrorResponse(BaseModel):
    """에러 응답"""
    meta: ResponseMeta
    errors: List[ErrorDetail] = Field(default_factory=list, description="에러 목록")