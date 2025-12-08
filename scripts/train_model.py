import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ---------------------------------------------------------
# 1. 프로젝트 경로 및 폰트 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.data_processor import load_and_engineer_features

# 한글 폰트 설정
import platform

if platform.system() == 'Darwin':
    font_family = 'AppleGothic'
elif platform.system() == 'Windows':
    font_family = 'Malgun Gothic'
else:
    font_family = 'NanumGothic'
plt.rc('font', family=font_family)
plt.rc('axes', unicode_minus=False)


def train_and_save_model():
    print("\n" + "=" * 60)
    print("[Start] 전세사기 위험도 예측 모델 학습 시작")
    print("=" * 60)

    # ---------------------------------------------------------
    # 2. 데이터 로드
    # ---------------------------------------------------------
    print("\n>> 1. 데이터 로드 및 전처리 중...")
    # data_processor에서 가공된 데이터(가짜 빚이 제거된 클린 데이터) 로드
    df = load_and_engineer_features()

    if df.empty:
        print("❌ 학습할 데이터가 없습니다.")
        return

    # ---------------------------------------------------------
    # 3. 추가 전처리 (One-Hot Encoding)
    # ---------------------------------------------------------
    # simple_type (APT, VILLA 등)을 원-핫 인코딩 -> type_APT, type_VILLA 생성
    if 'simple_type' in df.columns:
        df = pd.get_dummies(df, columns=['simple_type'], prefix='type')

    # ---------------------------------------------------------
    # 4. [핵심 수정] 정답지(Label) 생성: 'is_fraud'
    # ---------------------------------------------------------
    # 데이터가 'Clean'해졌으므로, 라벨링 기준을 '잠재적 위험'까지 포함하도록 넓혀야 함

    # 조건 1: HUG 보증보험 가입 불가 (전세가율이 너무 높음)
    cond_hug = df['hug_risk_ratio'] > 1.0

    # 조건 2: 깡통전세 위험군 (전세가율 80% 이상) - 통상적 위험 기준
    # (이전에는 가짜 빚을 더해서 1.0을 넘겼지만, 이제는 순수 전세가율이므로 0.8로 잡음)
    cond_debt = df['total_risk_ratio'] >= 0.8

    # 조건 3: 복합 위험군 (전세가율 70% 이상이면서 + 정성적 위험 점수(estimated_loan_ratio)가 높음)
    # 즉, 시세 대비 전세가 70%인데 집주인이 신탁이거나 건물이 근생이면 위험으로 간주
    cond_complex = (df['total_risk_ratio'] >= 0.7) & (df['estimated_loan_ratio'] >= 0.3)

    # 2) 정성적 위험 반영
    # "전세가율이 70%만 넘어도(0.7), 집주인이 신탁(is_trust_owner)이면 위험하다"라고 가르침
    cond_trust_risk = (df['total_risk_ratio'] >= 0.7) & (df['is_trust_owner'] == 1)

    # "전세가율이 70%만 넘어도, 집주인이 단기 투기꾼(short_term)이면 위험하다"
    cond_short_term = (df['total_risk_ratio'] >= 0.7) & (df['short_term_weight'] > 0)

    # 조건 4: 위반건축물
    cond_illegal = df['is_illegal'] == 1

    df['is_fraud'] = (cond_trust_risk | cond_short_term |cond_hug | cond_debt | cond_complex | cond_illegal).astype(int)

    total_cnt = len(df)
    fraud_cnt = df['is_fraud'].sum()
    safe_cnt = total_cnt - fraud_cnt

    print(f"\n[데이터 분포 확인]")
    print(f"   전체 데이터: {total_cnt}건")
    print(f"   위험(Fraud) 레이블: {fraud_cnt}건 ({fraud_cnt / total_cnt * 100:.1f}%)")
    print(f"   안전(Safe)  레이블: {safe_cnt}건")

    if fraud_cnt < 10 or safe_cnt < 10:
        print("⚠️ 경고: 데이터 불균형이 너무 심하거나 샘플이 부족하여 학습 효과가 낮을 수 있습니다.")

    # ---------------------------------------------------------
    # 5. 학습용 데이터셋 분리
    # ---------------------------------------------------------
    # 학습에 사용할 피처 정의
    # 주의: total_risk_ratio(전세가율) 자체가 정답을 만드는 핵심 변수이지만,
    # AI가 '어느 정도 비율일 때 위험한지' 경계선을 배우게 하기 위해 포함합니다.
    feature_candidates = [
        'jeonse_ratio',  # 전세가율
        'hug_risk_ratio',  # HUG 가입여부 지표
        'total_risk_ratio',  # 깡통전세율
        'estimated_loan_ratio',  # 정성적 위험 점수 (집주인/건물특성)
        'building_age',  # 건물 연식
        'is_illegal',  # 위반 여부
        'parking_per_household',  # 주차 (있다면)
        'is_micro_complex',  # 나홀로 아파트 여부 (있다면)
        'is_trust_owner',  #
        'short_term_weight'  #
    ]

    # 생성된 원-핫 인코딩 컬럼들 추가 (type_APT, type_VILLA ...)
    feature_candidates.extend([c for c in df.columns if c.startswith('type_')])

    # 실제 데이터프레임에 존재하는 컬럼만 최종 선택
    feature_cols = [f for f in feature_candidates if f in df.columns]

    print(f"   사용된 피처({len(feature_cols)}개): {feature_cols}")

    X = df[feature_cols]
    y = df['is_fraud']

    # 데이터가 너무 적으면 분리 없이 전체 학습 (테스트용)
    if len(df) < 50:
        print("⚠️ 데이터가 50건 미만입니다. train/test 분리 없이 전체 데이터로 학습합니다.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        if len(np.unique(y)) < 2:
            print("❌ 레이블 클래스가 1개뿐입니다 (모두 안전 or 모두 위험). 학습 불가.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # ---------------------------------------------------------
    # 6. 모델 학습 (Random Forest)
    # ---------------------------------------------------------
    print("\n>> 2. 모델 학습 수행 (Random Forest)...")
    rf_model = RandomForestClassifier(
        n_estimators=200,  # 트리 개수 늘림
        max_depth=10,
        min_samples_leaf=2,  # 과적합 방지
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("   학습 완료!")

    # ---------------------------------------------------------
    # 7. 성능 평가
    # ---------------------------------------------------------
    print("\n>> 3. 성능 평가 결과")
    y_pred = rf_model.predict(X_test)

    # ROC-AUC (클래스 2개 이상일 때만)
    roc = 0.0
    try:
        if len(np.unique(y_test)) > 1:
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_pred_proba)
    except Exception as e:
        print(f"   (ROC 계산 건너뜀: {e})")

    acc = accuracy_score(y_test, y_pred)

    print(f"   정확도(Accuracy): {acc:.4f}")
    print(f"   ROC-AUC 점수: {roc:.4f}")

    # ---------------------------------------------------------
    # 8. 결과 저장
    # ---------------------------------------------------------
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1) 모델 파일 저장
    model_path = os.path.join(model_dir, 'fraud_rf_model.pkl')
    joblib.dump(rf_model, model_path)
    print(f"\n>> 4. 모델 저장 완료: {model_path}")

    # 2) 피처 중요도 시각화
    try:
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # 상위 10개만 표시
        top_n = min(10, len(feature_cols))

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices][:top_n], y=np.array(feature_cols)[indices][:top_n], palette='viridis')
        plt.title("AI 모델 중요 변수 (Top Factors)")
        plt.xlabel("중요도 (Importance)")
        plt.tight_layout()

        plot_path = os.path.join(model_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        print(f"   -> 그래프 저장 완료: {plot_path}")
    except Exception as e:
        print(f"   (그래프 저장 실패: {e})")

    print("\n" + "=" * 60)
    print("✅ 학습 종료. 이제 predict.py를 실행할 수 있습니다.")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_model()