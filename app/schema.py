from pydantic import BaseModel, Field
from typing import Literal, Optional


# ---------------------------------------------------------
# 1. 입력 스키마 (Request)
# 사용자가 API에 보낼 데이터 구조입니다.
# ---------------------------------------------------------
class PredictionRequest(BaseModel):
    address: str = Field(
        ...,
        description="분석할 매물의 도로명 또는 지번 주소 (예: '인천광역시 부평구 산곡동 145')"
    )
    deposit: int = Field(
        ...,
        description="계약하려는 전세 보증금 액수 (단위: 원, 예: 200000000)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "address": "인천광역시 부평구 산곡동 145",
                "deposit": 20000
            }
        }


# ---------------------------------------------------------
# 2. 출력 스키마 (Response)
# API가 분석 후 사용자에게 돌려줄 결과 데이터 구조입니다.
# ---------------------------------------------------------

class RiskDetails(BaseModel):
    """위험도 분석 상세 항목"""
    hug_ratio: float = Field(
        ...,
        description="HUG 전세보증보험 가입 한도 대비 전세가 비율 (%, 100% 초과 시 가입 불가)"
    )
    total_ratio: float = Field(
        ...,
        description="깡통전세 위험도: (추정 대출금 + 전세금) / 추정 시세 비율 (%, 100% 초과 시 위험)"
    )
    is_trust: bool = Field(
        ...,
        description="신탁 등기 여부 (True: 신탁사 소유로 주의 필요, False: 일반 매물)"
    )
    is_short_term: bool = Field(
        ...,
        description="집주인 단기 소유 여부 (True: 소유권 이전 2년 미만, False: 장기 보유)"
    )


class PredictionResponse(BaseModel):
    """최종 예측 결과 응답"""
    address: str = Field(..., description="요청받은 주소")
    building_name: str = Field(..., description="건축물대장상 건물명 및 동/호수")
    deposit: str = Field(..., description="입력된 전세 보증금 (단위 변환됨)")

    risk_score: float = Field(
        ...,
        description="AI 모델이 예측한 전세사기 위험 확률 (0 ~ 100점)"
    )
    risk_level: Literal['SAFE', 'RISKY'] = Field(
        ...,
        description="위험 등급 (SAFE: 안전, RISKY: 위험)"
    )

    details: RiskDetails = Field(..., description="상세 분석 지표")

    class Config:
        json_schema_extra = {
            "example": {
                "address": "인천광역시 부평구 산곡동 145",
                "building_name": "311동 101호",
                "deposit": "20000만원",
                "risk_score": 71.0,
                "risk_level": "RISKY",
                "details": {
                    "hug_ratio": 95.6,
                    "total_ratio": 106.1,
                    "is_trust": False,
                    "is_short_term": False
                }
            }
        }