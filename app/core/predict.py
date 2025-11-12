import torch
import torch.nn.functional as F
from joblib import load
import pandas as pd
import numpy as np
import os

# 현재 파일 위치 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(BASE_DIR, '..', '..', 'assets')
MODEL_PATH = os.path.join(ASSETS_PATH, 'fraud_model.pth')
SCALER_PATH = os.path.join(ASSETS_PATH, 'scaler.pkl')
COLUMNS_PATH = os.path.join(ASSETS_PATH, 'feature_columns.txt')

# FastAPI 앱 전역에서 사용할 객체들
ml_assets = {}

# Pydantic 스키마 (schema.py)
from app.schema import PropertyFeatures, PredictionResponse
# PyTorch 모델 (model_def.py)
from app.model_def import FraudNet

def load_ml_assets():
    """
    FastAPI 앱 시작 시 한 번만 호출되어,
    모델, 스케일러, 컬럼 목록을 메모리에 로드합니다.
    """
    print("--- 머신러닝 에셋 로딩 시작 ---")
    try:
        # 1. 학습에 사용된 컬럼 목록 로드 (매우 중요!)
        with open(COLUMNS_PATH, 'r') as f:
            ml_assets['feature_columns'] = f.read().splitlines()

        NUM_FEATURES = len(ml_assets['feature_columns'])
        print(f"컬럼 목록 로드 완료 (총 {NUM_FEATURES}개)")

        # 2. PyTorch 모델 로드
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FraudNet(input_size=NUM_FEATURES, num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()  # 추론 모드로 설정

        ml_assets['model'] = model
        ml_assets['device'] = device
        print(f"PyTorch 모델 로드 완료 (Device: {device})")

        # 3. 스케일러 로드
        ml_assets['scaler'] = load(SCALER_PATH)
        print("StandardScaler 로드 완료")

        print("--- 모든 에셋 로딩 성공 ---")

    except FileNotFoundError as e:
        print(f"에러: {e}")
        print("먼저 `scripts/train_model.py`를 실행하여 에셋 파일을 생성해야 합니다.")
        raise
    except Exception as e:
        print(f"에셋 로딩 중 예기치 않은 에러 발생: {e}")
        raise


def preprocess_input(features: PropertyFeatures) -> np.ndarray:
    """
    API로 받은 Pydantic 객체를 모델이 이해할 수 있는
    스케일링된 numpy 배열로 변환합니다.
    """
    # 1. Pydantic 객체를 딕셔너리로, 다시 DataFrame으로 변환
    input_data = features.model_dump()
    df = pd.DataFrame([input_data])  # 1줄짜리 DataFrame

    # 2. 원-핫 인코딩
    # (주의) 학습 시 사용한 카테고리 순서와 동일해야 함
    categories = ['아파트', '다세대주택', '오피스텔', '근린생활시설']
    df['building_use'] = pd.Categorical(df['building_use'], categories=categories)
    df = pd.get_dummies(df, columns=['building_use'], drop_first=False)

    # 3. (핵심) 학습 시 사용한 컬럼 순서/목록으로 재정렬
    # API로 들어온 데이터에 없는 컬럼(e.g., building_use_아파트)은 0으로 채워짐
    trained_columns = ml_assets['feature_columns']
    df = df.reindex(columns=trained_columns, fill_value=0)

    # 4. 스케일링 (fit이 아닌 transform 사용)
    scaler = ml_assets['scaler']
    scaled_data = scaler.transform(df)

    return scaled_data


def predict_risk(features: PropertyFeatures) -> PredictionResponse:
    """
    API 엔드포인트에서 호출하는 메인 예측 함수
    """
    # 1. 입력 데이터 전처리
    processed_data = preprocess_input(features)

    # 2. PyTorch 텐서로 변환
    tensor_data = torch.tensor(processed_data, dtype=torch.float32).to(ml_assets['device'])

    # 3. 모델 추론 (그래디언트 계산 X)
    with torch.no_grad():
        outputs = ml_assets['model'](tensor_data)

        # Softmax를 적용하여 확률 계산
        # outputs: [정상 확률, 위험 확률]
        probabilities = F.softmax(outputs, dim=1)

        # '위험(1)'할 확률
        risk_probability = probabilities[0, 1].item()

        # 4. 결과 포맷팅
    if risk_probability > 0.5:  # (임계값은 비즈니스 로직에 따라 조절)
        risk_level = "RISKY"
    else:
        risk_level = "NORMAL"

    return PredictionResponse(
        risk_level=risk_level,
        probability=round(risk_probability, 4)
    )