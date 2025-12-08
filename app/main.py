from fastapi import FastAPI, HTTPException
from app.schema import PredictionRequest, PredictionResponse
from app.core.predict import predict_risk

app = FastAPI(
    title="전세 사기 위험도 예측 API",
    description="등기부등본, 건축물대장 등의 주요 특성을 입력받아 해당 매물의 위험도를 예측합니다.",
    version="1.0.0"
)

@app.get("/", summary="API 상태 확인")
def read_root():
    return {"status": "Healthy", "service": "Fraud Detector AI"}


@app.post("/predict",
          summary="전세사기 위험도 진단",
          description="주소와 전세보증금을 입력하면 AI가 위험도를 분석합니다.",
          response_model=PredictionResponse)
async def post_predict(features: PredictionRequest):
    """
    - **address**: 등기부등본상 도로명/지번 주소 (예: 인천 부평구 산곡동 145)
    - **deposit**: 전세 보증금 액수 (단위: 만원, 예: 20000)
    """
    result = predict_risk(features.address, features.deposit)

    # 에러 처리 (주소를 못 찾거나 데이터가 없는 경우)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result