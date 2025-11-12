from pydantic import BaseModel
from typing import Literal


# API가 입력받을 JSON 형태 정의
class PropertyFeatures(BaseModel):
    jeonse_ratio: float
    loan_plus_jeonse_ratio: float
    has_trust: int  # 0 또는 1
    is_illegal_building: int  # 0 또는 1

    # 학습 시 사용한 텍스트 그대로 받음
    building_use: Literal['아파트', '다세대주택', '오피스텔', '근린생활시설']

    owner_changed_recently: int  # 0 또는 1

    # 모델에 사용할 수 있는 예시 데이터
    class Config:
        json_schema_extra = {
            "example": {
                "jeonse_ratio": 0.95,
                "loan_plus_jeonse_ratio": 1.1,
                "has_trust": 1,
                "is_illegal_building": 0,
                "building_use": "오피스텔",
                "owner_changed_recently": 1
            }
        }


# API가 반환할 JSON 형태 정의
class PredictionResponse(BaseModel):
    risk_level: Literal['NORMAL', 'RISKY']
    probability: float  # 위험할 확률 (0.0 ~ 1.0)