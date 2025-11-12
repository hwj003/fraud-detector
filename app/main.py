from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from app.schema import PropertyFeatures, PredictionResponse
from app.core.predict import load_ml_assets, predict_risk
app = FastAPI()


# FastAPI 3.10+ 권장 방식
# 앱 시작 시 load_ml_assets 호출, 종료 시 정리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작
    load_ml_assets()
    yield
    # 종료
    # ml_assets.clear() (필요하다면)

app = FastAPI(
    title="전세 사기 위험도 예측 API",
    description="등기부등본, 건축물대장 등의 주요 특성을 입력받아 해당 매물의 위험도를 예측합니다.",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/", summary="API 상태 확인")
def read_root():
    return {"status": "Fraud Detector API is running!"}

@app.post("/predict",
          summary="매물 위험도 예측",
          response_model=PredictionResponse)
async def post_predict(features: PropertyFeatures):
    """
    매물의 주요 특성(JSON)을 입력받아 위험도('RISKY'/'NORMAL')와
    위험 확률(probability)을 반환합니다.

    - **jeonse_ratio**: 전세가율 (e.g., 0.9)
    - **loan_plus_jeonse_ratio**: (선순위부채+전세금)/매매가 (e.g., 1.1)
    - **has_trust**: 신탁등기 여부 (1: 있음, 0: 없음)
    - **is_illegal_building**: 위반건축물 여부 (1: 있음, 0: 없음)
    - **building_use**: 건물 용도 (e.g., '오피스텔')
    - **owner_changed_recently**: 최근 소유주 변경 여부 (1: 있음, 0: 없음)
    """
    prediction = predict_risk(features)
    return prediction