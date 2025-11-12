# fraud_detector_project/app/core/predict.py

import torch
import torch.nn.functional as F
from joblib import load
import pandas as pd
import numpy as np
import os

# [신규] DB 연결 엔진, SQL 텍스트, 스키마 임포트
from app.schema import PropertyFeatures, PredictionResponse
from app.model_def import FraudNet

# --- 에셋 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(BASE_DIR, '..', '..', 'assets')
MODEL_PATH = os.path.join(ASSETS_PATH, 'fraud_model.pth')
SCALER_PATH = os.path.join(ASSETS_PATH, 'scaler.pkl')
COLUMNS_PATH = os.path.join(ASSETS_PATH, 'feature_columns.txt')

ml_assets = {}


def load_ml_assets():
    """FastAPI 앱 시작 시 모델, 스케일러, 컬럼 목록을 로드합니다."""

    print("--- 머신러닝 에셋 로딩 시작 ---")
    try:
        with open(COLUMNS_PATH, 'r', encoding='utf-8') as f:
            ml_assets['feature_columns'] = f.read().splitlines()
        NUM_FEATURES = len(ml_assets['feature_columns'])
        print(f"컬럼 목록 로드 완료 (총 {NUM_FEATURES}개)")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FraudNet(input_size=NUM_FEATURES, num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        ml_assets['model'] = model
        ml_assets['device'] = device
        print(f"PyTorch 모델 로드 완료 (Device: {device})")

        ml_assets['scaler'] = load(SCALER_PATH)
        print("StandardScaler 로드 완료")
        print("--- 모든 에셋 로딩 성공 ---")
    except Exception as e:
        print(f"에셋 로딩 중 예기치 않은 에러 발생: {e}")
        raise


def preprocess_and_engineer(features: PropertyFeatures) -> np.ndarray:
    """
    사용자 입력(features)을 받아 실시간 원-핫 인코딩 수행 후,
    스케일링하여 모델 입력용 벡터를 반환합니다.
    """

    # 1. Pydantic 객체를 딕셔너리로 변환
    feature_vector_dict = features.model_dump()

    # 2. 'building_use' 문자열을 원-핫 인코딩으로 변환
    # (data_processor.py의 로직과 100% 동일해야 함)

    # (1) 딕셔너리에서 'building_use' 문자열 값을 꺼냄
    building_use_category = feature_vector_dict.pop('building_use', '기타')

    categories = ['아파트', '다세대주택', '오피스텔', '근린생활시설', '기타']

    # 딕셔너리에 원-핫 인코딩된 컬럼 추가
    for cat in categories:
        col_name = f'building_use_{cat}'
        feature_vector_dict[col_name] = 1 if building_use_category == cat else 0

    trained_columns = ml_assets['feature_columns']

    df = pd.DataFrame([feature_vector_dict])
    df = df.reindex(columns=trained_columns, fill_value=0)

    # 4. 스케일링
    scaler = ml_assets['scaler']
    scaled_data = scaler.transform(df)

    # 5. nan/inf 방지
    scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=5.0, neginf=-5.0)

    return scaled_data


# 메인 예측 함수
def predict_risk(features: PropertyFeatures) -> PredictionResponse:
    """ API 엔드포인트에서 호출하는 메인 예측 함수 """

    processed_data = preprocess_and_engineer(features)
    tensor_data = torch.tensor(processed_data, dtype=torch.float32).to(ml_assets['device'])

    with torch.no_grad():
        outputs = ml_assets['model'](tensor_data)
        probabilities = F.softmax(outputs, dim=1)
        risk_probability = probabilities[0, 1].item()

        # [수정] 모델 출력이 nan인 경우 예외 처리
    if np.isnan(risk_probability):
        print("[경고] 모델 출력이 NaN입니다. 입력 특성에 여전히 문제가 있습니다.")
        risk_probability = 0.5  # (중립값 반환)
        risk_level = "NORMAL"  # (또는 "RISKY" - 비즈니스 정책 필요)
    else:
        risk_level = "RISKY" if risk_probability > 0.5 else "NORMAL"

    return PredictionResponse(
        risk_level=risk_level,
        probability=round(risk_probability, 4)
    )