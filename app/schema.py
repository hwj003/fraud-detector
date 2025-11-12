from pydantic import BaseModel, Field
from typing import Literal


class PredictionResponse(BaseModel):
    risk_level: Literal['NORMAL', 'RISKY']
    probability: float


class PropertyFeatures(BaseModel):
    jeonse_ratio: float
    is_illegal_building: int
    building_age: float

    building_use: Literal['아파트', '다세대주택', '오피스텔', '근린생활시설', '기타']

    has_trust: int
    loan_plus_jeonse_ratio: float

    class Config:
        json_schema_extra = {
            "example": {
                "jeonse_ratio": 0.46,
                "is_illegal_building": 0,
                "building_age": 34.0,
                "building_use": "아파트",
                "has_trust": 0,
                "loan_plus_jeonse_ratio": 0.46
            }
        }