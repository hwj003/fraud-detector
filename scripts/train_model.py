# fraud_detector_project/scripts/train_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from joblib import dump  # 스케일러 저장을 위해 joblib 사용
import os
import sys

# Snorkel
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe

# 프로젝트 루트 경로를 sys.path에 추가 (app.model_def 임포트 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# FastAPI 앱의 'app/model_def.py'에서 모델 정의를 가져옴 (코드 중복 방지)
from app.model_def import FraudNet

# --- 상수 정의 ---
FRAUD = 1
NORMAL = 0
ABSTAIN = -1
ASSETS_DIR = os.path.join(project_root, 'assets')
MODEL_PATH = os.path.join(ASSETS_DIR, 'fraud_model.pth')
SCALER_PATH = os.path.join(ASSETS_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(ASSETS_DIR, 'feature_columns.txt')

# assets 폴더 생성
os.makedirs(ASSETS_DIR, exist_ok=True)


# --- [1] 데이터 준비 (시뮬레이션) ---
def create_dummy_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'jeonse_ratio': np.random.uniform(0.6, 1.2, num_samples),
        'loan_plus_jeonse_ratio': np.random.uniform(0.5, 1.5, num_samples),
        'has_trust': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'is_illegal_building': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        'building_use': np.random.choice(  # API는 이 텍스트 값을 받을 것임
            ['아파트', '다세대주택', '오피스텔', '근린생활시설'],
            num_samples,
            p=[0.4, 0.3, 0.2, 0.1]
        ),
        'owner_changed_recently': np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)

    # (중요) API에서 재현하기 위해 원-핫 인코딩 수행
    # API에서는 '아파트' 1개만 들어오므로, 카테고리를 고정해야 함
    categories = ['아파트', '다세대주택', '오피스텔', '근린생활시설']
    df['building_use'] = pd.Categorical(df['building_use'], categories=categories)
    df = pd.get_dummies(df, columns=['building_use'], drop_first=False)

    return df


df_unlabeled = create_dummy_data(num_samples=1000)
print("--- [1] 가상 특성 데이터 생성 (원-핫 인코딩 완료) ---")
print(df_unlabeled.head())

# (중요) API 서빙 시 동일한 순서의 컬럼을 사용하기 위해 컬럼 리스트 저장
final_feature_columns = list(df_unlabeled.columns)
with open(COLUMNS_PATH, 'w') as f:
    f.write('\n'.join(final_feature_columns))
print(f"--- 학습에 사용된 최종 컬럼 (총 {len(final_feature_columns)}개) 저장 완료 ---")

NUM_FEATURES = len(final_feature_columns)


# --- [2] 약한 라벨링 (Snorkel) ---
def lf_trust_property(x):
    """LF 1: 신탁등기는 매우 강력한 위험 신호"""
    return FRAUD if x.has_trust == 1 else ABSTAIN

def lf_high_debt_ratio(x):
    """LF 2: (선순위부채 + 전세금)이 매매가보다 높으면 위험"""
    return FRAUD if x.loan_plus_jeonse_ratio > 1.0 else ABSTAIN

def lf_commercial_use(x):
    """LF 4: '근린생활시설' 매물은 위험"""
    # 'building_use_근린생활시설' 컬럼이 있는지 확인 (원-핫 인코딩)
    if 'building_use_근린생활시설' in x and x['building_use_근린생활시설'] == 1:
        return FRAUD
    return ABSTAIN

def lf_safe_apartment(x):
    """LF 5: (안전 신호) 아파트이고 부채비율이 낮으면 정상"""
    if 'building_use_아파트' in x and x['building_use_아파트'] == 1 and x.loan_plus_jeonse_ratio < 0.7:
        return NORMAL
    return ABSTAIN

# [수정 2] LFs 리스트를 만들 때 LabelingFunction 객체로 감싸줍니다.
#          이때 name과 f(함수)를 명시적으로 전달합니다.
lfs = [
    LabelingFunction(name="lf_trust_property", f=lf_trust_property),
    LabelingFunction(name="lf_high_debt_ratio", f=lf_high_debt_ratio),
    LabelingFunction(name="lf_commercial_use", f=lf_commercial_use),
    LabelingFunction(name="lf_safe_apartment", f=lf_safe_apartment)
]

# Pandas DataFrame에 LFs 적용
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_unlabeled) # (1000, 4) 크기의 투표 행렬 (함수 4개)

print("\n--- [2-1] LFs 투표 행렬 (L_train) 생성 (상위 5개) ---")
print(L_train[:5])

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, lr=0.001, log_freq=500)
probs_train = label_model.predict_proba(L=L_train)
weak_labels = np.argmax(probs_train, axis=1)

df_train_filtered, labels_train_filtered = filter_unlabeled_dataframe(
    X=df_unlabeled, y=weak_labels, L=L_train
)
print(f"--- [2] 약한 라벨링 완료 (학습 데이터 {len(df_train_filtered)}개) ---")

# --- [3] PyTorch 모델 학습 ---

# 3-1. 데이터 전처리 (스케일러 Fit)
X = df_train_filtered.values
y = labels_train_filtered

# (중요) 이 스케일러를 저장해야 함
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3-2. 모델 정의 및 손실 함수
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FraudNet(input_size=NUM_FEATURES, num_classes=2).to(device)

# 불균형 데이터 가중치
counts = pd.Series(y).value_counts()
weights = (counts.sum() / counts).values
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3-3. 학습 루프
print("--- [3] PyTorch 모델 학습 시작 ---")
num_epochs = 20
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("--- 모델 학습 완료 ---")

# --- [4] 산출물 저장 ---

# 1. 모델 가중치 저장
torch.save(model.state_dict(), MODEL_PATH)
print(f"모델 저장 완료: {MODEL_PATH}")

# 2. 스케일러 저장
dump(scaler, SCALER_PATH)
print(f"스케일러 저장 완료: {SCALER_PATH}")

print("\n=== 모든 학습 및 산출물 저장이 완료되었습니다. API를 실행하세요. ===")