import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib # 모델 저장용 라이브러리

# Scikit-Learn 머신러닝 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 한글 폰트 설정 (그래프용)
import platform
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
else: plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# ---------------------------------------------------------
# 1. 프로젝트 경로 설정
# ---------------------------------------------------------
# 스크립트 파일 위치 기준 상위 폴더를 루트로 지정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 데이터 프로세서 임포트
from scripts.data_processor import load_and_engineer_features

# 한글 폰트 설정 (그래프 저장용)
import platform
if platform.system() == 'Darwin': font_family = 'AppleGothic'
elif platform.system() == 'Windows': font_family = 'Malgun Gothic'
else: font_family = 'NanumGothic'
plt.rc('font', family=font_family)
plt.rc('axes', unicode_minus=False)

def train_and_save_model():
    print("\n" + "=" * 60)
    print("🚀 [Start] 전세사기 위험도 예측 모델 학습 시작")
    print("=" * 60)

    # ---------------------------------------------------------
    # 2. 데이터 로드 및 라벨링
    # ---------------------------------------------------------
    print("\n>> 1. 데이터 로드 및 전처리 중...")
    df = load_and_engineer_features()

    # [약한 라벨링] 정답지(is_fraud) 생성
    # 기준: HUG 불가 OR 깡통전세 OR 신탁 OR 단기소유(동시진행)
    df['is_fraud'] = (
            (df['hug_risk_ratio'] > 1.0) |
            (df['total_risk_ratio'] > 1.0) |
            (df['is_trust_owner'] == 1) |
            (df['short_term_weight'] >= 0.3) |
            (df['is_illegal_building'] == 1)
    ).astype(int)

    total_cnt = len(df)
    fraud_cnt = df['is_fraud'].sum()
    safe_cnt = total_cnt - fraud_cnt

    print(f"   전체 데이터: {total_cnt}건")
    print(f"   위험(Fraud) 클래스: {fraud_cnt}건 ({fraud_cnt / total_cnt * 100:.1f}%)")
    print(f"   안전(Safe) 클래스: {safe_cnt}건")

    # ---------------------------------------------------------
    # 3. 학습용 데이터셋 분리
    # ---------------------------------------------------------
    # 학습에 사용할 피처 정의
    feature_cols = [
        'jeonse_ratio',  # 전세가율
        'hug_risk_ratio',  # HUG 기준 위험도
        'total_risk_ratio',  # 깡통전세 위험도
        'building_age',  # 건물 연식
        'parking_per_household',  # 세대당 주차대수
        'is_micro_complex',  # 나홀로 아파트 여부
        'estimated_loan_ratio',  # 추정 대출 비율
        'is_trust_owner',  # 신탁 여부
        'short_term_weight',  # 단기 소유 위험도
        'is_illegal_building', # 위반 건축물 여부
    ]

    # One-Hot Encoding된 용도 컬럼들 추가 (use_아파트 등, use_apr_day는 날짜이므로 제외!)
    feature_cols.extend([
        c for c in df.columns
        if c.startswith('use_') and c != 'use_apr_day'
    ])

    # 실제 데이터프레임에 존재하는 컬럼만 선택 (에러 방지)
    feature_cols = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols]
    y = df['is_fraud']

    # 8:2 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 4. 모델 학습 (Random Forest)
    # ---------------------------------------------------------
    print("\n>> 3. 모델 학습 수행 (Random Forest)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("   학습 완료!")

    # ---------------------------------------------------------
    # 5. 성능 평가
    # ---------------------------------------------------------
    print("\n>> 4. 성능 평가 결과")
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)

    print(f"   정확도(Accuracy): {acc:.4f}")
    print(f"   ROC-AUC 점수: {roc:.4f}")
    print("\n   [상세 리포트]")
    print(classification_report(y_test, y_pred, target_names=['안전(0)', '위험(1)']))

    # ---------------------------------------------------------
    # 6. 결과 저장 (모델 & 피처 중요도 그래프)
    # ---------------------------------------------------------
    # 저장 경로 설정
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1) 모델 파일 저장
    model_path = os.path.join(model_dir, 'fraud_rf_model.pkl')
    joblib.dump(rf_model, model_path)
    print(f"\n>> 5. 모델 저장 완료: {model_path}")

    # 2) 피처 중요도 이미지 저장
    print("   -> 피처 중요도 그래프 저장 중...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=X_train.columns[indices], palette='viridis')
    plt.title("전세사기 예측 모델 중요 변수 (Feature Importance)")
    plt.xlabel("중요도 (Importance Score)")
    plt.ylabel("변수명")
    plt.tight_layout()

    plot_path = os.path.join(model_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    print(f"   -> 그래프 저장 완료: {plot_path}")

    print("\n" + "=" * 60)
    print("🎉 모든 학습 과정이 성공적으로 끝났습니다.")
    print("=" * 60)

if __name__ == "__main__":
    train_and_save_model()