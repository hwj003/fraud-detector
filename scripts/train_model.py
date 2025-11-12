# fraud_detector_project/scripts/train_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os
import sys

# Snorkel
from snorkel.labeling import LabelingFunction, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe

# --- [필수] 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# --- 중앙 설정 및 모델 정의 임포트 ---
# (이 시점에 app/core/config.py가 실행되며 DB 연결 로그가 찍힐 수 있습니다)
from app.core.config import engine
from app.model_def import FraudNet
from scripts.data_processor import load_and_engineer_features

# --- 상수 정의 ---
FRAUD = 1
NORMAL = 0
ABSTAIN = -1
ASSETS_DIR = os.path.join(project_root, 'assets')
MODEL_PATH = os.path.join(ASSETS_DIR, 'fraud_model.pth')
SCALER_PATH = os.path.join(ASSETS_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(ASSETS_DIR, 'feature_columns.txt')


def lf_high_debt_ratio(x):
    """LF 2: 부채+전세가율(가상)이 100% 이상이면 위험"""
    return FRAUD if x['loan_plus_jeonse_ratio'] > 1.0 else ABSTAIN

def lf_high_jeonse_ratio_only(x):
    """LF 3: (부채 없는) 전세가율이 90% 이상이면 위험"""
    return FRAUD if x['jeonse_ratio'] > 0.9 else ABSTAIN

def lf_illegal_and_high_ratio(x):
    """LF 4: 위반건축물이면서 전세가율이 80% 이상이면 위험"""
    return FRAUD if x['is_illegal_building'] == 1 and x['jeonse_ratio'] > 0.8 else ABSTAIN

def lf_commercial_use(x):
    """LF 5: '근린생활시설' 매물은 위험"""
    return FRAUD if x['building_use_근린생활시설'] == 1 else ABSTAIN

def lf_safe_apartment(x):
    """LF 6: (안전 신호) 아파트이고 부채+전세가율이 70% 미만이면 정상"""
    if x['building_use_아파트'] == 1 and x['loan_plus_jeonse_ratio'] < 0.7:
        return NORMAL
    return ABSTAIN

# [추가 LF 7] - 아파트 조건 없이, 부채 포함 비율이 낮으면 정상
def lf_low_debt_ratio(x):
    """LF 7: (안전 신호) 부채+전세가율이 70% 미만이면 정상"""
    if x['loan_plus_jeonse_ratio'] < 0.7:
        return NORMAL
    return ABSTAIN

# [추가 LF 8] - 전세가율 자체가 매우 낮으면 정상
def lf_very_low_jeonse(x):
    """LF 8: (안전 신호) 전세가율이 60% 미만이면 정상"""
    if x['jeonse_ratio'] < 0.6:
        return NORMAL
    return ABSTAIN


def main():
    # assets 폴더 생성
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # --- [1단계] 데이터 준비 (DB에서 실제 데이터 로드 및 가공) ---
    print("--- [1] data_processor.py 호출 (특성 공학 시작) ---")
    df_unlabeled = load_and_engineer_features()
    print("--- [1] 특성 공학 완료 ---")

    final_feature_columns = list(df_unlabeled.columns)
    with open(COLUMNS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_feature_columns))
    print(f"--- 학습에 사용된 최종 컬럼 (총 {len(final_feature_columns)}개) 저장 완료 ---")
    NUM_FEATURES = len(final_feature_columns)

    # --- [2단계] 약한 라벨링 (Snorkel) ---
    print("--- [2] 약한 라벨링(Snorkel) 시작 ---")

    lfs = [
        LabelingFunction(name="lf_high_debt_ratio", f=lf_high_debt_ratio),
        LabelingFunction(name="lf_high_jeonse_ratio_only", f=lf_high_jeonse_ratio_only),
        LabelingFunction(name="lf_illegal_and_high_ratio", f=lf_illegal_and_high_ratio),
        LabelingFunction(name="lf_commercial_use", f=lf_commercial_use),
        LabelingFunction(name="lf_safe_apartment", f=lf_safe_apartment),
        LabelingFunction(name="lf_low_debt_ratio", f=lf_low_debt_ratio),
        LabelingFunction(name="lf_very_low_jeonse", f=lf_very_low_jeonse)
    ]

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_unlabeled)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, lr=0.001, log_freq=100)

    probs_train = label_model.predict_proba(L=L_train)
    weak_labels = np.argmax(probs_train, axis=1)

    df_train_filtered, labels_train_filtered = filter_unlabeled_dataframe(
        X=df_unlabeled, y=weak_labels, L=L_train
    )

    print(f"--- [2] 약한 라벨링 완료 (총 {len(df_unlabeled)}개 중 {len(df_train_filtered)}개 라벨링 성공) ---")
    print("생성된 라벨 분포:\n", pd.Series(labels_train_filtered).value_counts())

    if len(df_train_filtered) == 0:
        print("[오류] 라벨링된 데이터가 0개입니다. LFs가 ABSTAIN(-1)만 반환했는지 확인하세요.")
        return  # 학습 중단

    # --- [3단계] PyTorch DataLoader 준비 ---
    print("--- [3] PyTorch DataLoader 준비 중... ---")

    X = df_train_filtered[final_feature_columns].values
    y = labels_train_filtered
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=5.0, neginf=-5.0)

    dump(scaler, SCALER_PATH)
    print(f"StandardScaler 저장 완료: {SCALER_PATH}")

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # --- [4단계] 모델, 손실 함수, 옵티마이저 정의 ---
    print("--- [4] 모델 및 옵티마이저 정의 ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FraudNet(input_size=NUM_FEATURES, num_classes=2).to(device)

    counts = pd.Series(y).value_counts()
    # (라벨이 하나만 있는 경우(e.g., 0만 있음) 에러 방지)
    if 0 not in counts: counts[0] = 1
    if 1 not in counts: counts[1] = 1

    weights = (counts.sum() / counts).values
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"클래스 가중치 (0:정상, 1:사기): {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- [5단계] 모델 학습 ---
    print("--- [5] PyTorch 모델 학습 시작 ---")
    num_epochs = 20
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    print("--- 모델 학습 완료 ---")

    # --- [6단계] 최종 모델 저장 ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n--- [성공] 학습된 모델 저장 완료 ---")
    print(f"모델 위치: {MODEL_PATH}")
    print("이제 'python run_api.py'를 실행하여 API 서버를 켤 수 있습니다.")


# --- [신규] 스크립트가 "직접 실행"되었을 때만 main() 함수 호출 ---
if __name__ == "__main__":
    main()