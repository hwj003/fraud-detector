import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime
from sqlalchemy import text
from app.services.feature_service import calculate_risk_features, convert_to_model_input
# ---------------------------------------------------------
# 1. 프로젝트 설정 로드
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from app.core import get_engine
engine = get_engine()


# ---------------------------------------------------------
# 2. 헬퍼 함수
# ---------------------------------------------------------
def _create_join_key_robust(row, col_map):
    try:
        sgg = str(row[col_map['sgg']]).strip()
        bjd = str(row[col_map['bjd']]).strip()
        bon_val = row[col_map['bon']]
        bu_val = row[col_map['bu']]

        if pd.isna(bon_val) or pd.isna(bu_val): return None
        bon_str = str(int(float(bon_val))).zfill(4)
        bu_str = str(int(float(bu_val))).zfill(4)

        return f"{sgg}-{bjd}-{bon_str}-{bu_str}"
    except Exception:
        return None


def _generate_key_from_pnu(unique_no):
    try:
        if pd.isna(unique_no): return None
        s = str(unique_no).replace("-", "").strip()
        if len(s) < 19: return None
        return f"{s[0:5]}-{s[5:10]}-{s[11:15]}-{s[15:19]}"
    except Exception:
        return None


# ---------------------------------------------------------
# 3. 메인 파이프라인
# ---------------------------------------------------------
def run_risk_analysis_pipeline():
    print(f"[{datetime.now()}] --- 1. 데이터 로드 시작 ---")

    # 1. 건축물대장
    sql_build = """
        SELECT b.id as building_info_id, b.unique_number, b.main_use, 
               b.owner_name, b.exclusive_area AS AREA, 
               b.is_violating_building, b.detail_address, b.lot_address,
               t.use_apr_day, b.ownership_changed_date
        FROM building_info b
        LEFT JOIN building_title_info t 
            ON SUBSTR(b.unique_number, 1, 19) = t.unique_number
        WHERE b.unique_number IS NOT NULL
    """

    # 2. 전세 실거래가
    sql_rent = """
        SELECT 시군구, 법정동, 본번, 부번, 
               보증금 as rent_price, 계약일 as contract_date, 
               전용면적 as rent_area
        FROM raw_rent
        WHERE 월세 = 0 AND 계약일 >= '20230101'
    """

    # 3. 매매 실거래가
    sql_trade = """
        SELECT 시군구, 법정동, 본번, 부번, 
               거래금액 as trade_price, 계약일 as trade_date
        FROM raw_trade
        WHERE 계약일 >= '20230101'
    """

    # 4. 공시가격
    sql_pub = """
        SELECT building_info_id, price as public_price, base_date as price_date
        FROM public_price_history ORDER BY base_date ASC
    """

    with engine.connect() as conn:
        df_build = pd.read_sql(sql_build, conn)
        df_rent = pd.read_sql(sql_rent, conn)
        df_trade = pd.read_sql(sql_trade, conn)
        df_pub = pd.read_sql(sql_pub, conn)

    print(f"-> 건축물대장 로드: {len(df_build)}건")

    # --- 전처리 ---
    for df in [df_rent, df_trade, df_build]:
        cols = ['rent_price', 'trade_price', 'rent_area', 'AREA']
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    df_pub['public_price'] = pd.to_numeric(df_pub['public_price'], errors='coerce') / 10000

    df_rent['contract_date'] = pd.to_datetime(df_rent['contract_date'], errors='coerce')
    df_trade['trade_date'] = pd.to_datetime(df_trade['trade_date'], errors='coerce')
    df_pub['price_date'] = pd.to_datetime(df_pub['price_date'], errors='coerce')
    df_build['use_apr_day'] = pd.to_datetime(df_build['use_apr_day'], errors='coerce')
    df_build['ownership_changed_date'] = pd.to_datetime(df_build['ownership_changed_date'], errors='coerce')

    col_map = {'sgg': '시군구', 'bjd': '법정동', 'bon': '본번', 'bu': '부번'}
    df_rent['key'] = df_rent.apply(lambda row: _create_join_key_robust(row, col_map), axis=1)
    df_trade['key'] = df_trade.apply(lambda row: _create_join_key_robust(row, col_map), axis=1)
    df_build['key'] = df_build['unique_number'].apply(_generate_key_from_pnu)

    df_rent = df_rent.dropna(subset=['key', 'rent_price'])
    df_build = df_build.dropna(subset=['key'])
    df_rent = df_rent.sort_values('contract_date', ascending=False).drop_duplicates(subset=['key'], keep='first')

    # --- 병합 ---
    df_rent = df_rent.sort_values('contract_date')
    df_trade = df_trade.sort_values('trade_date')

    df_merged = pd.merge_asof(
        df_rent, df_trade[['key', 'trade_price', 'trade_date']],
        left_on='contract_date', right_on='trade_date',
        by='key', direction='backward', tolerance=pd.Timedelta(days=365 * 2)
    )

    df_merged = pd.merge(df_merged, df_build, on='key', how='left')
    df_merged = df_merged.dropna(subset=['building_info_id'])

    df_merged['area_diff'] = abs(df_merged['rent_area'] - df_merged['AREA'])
    initial_cnt = len(df_merged)
    df_merged = df_merged[df_merged['area_diff'] < 3.3].copy()
    print(f"-> 면적 불일치 제거: {initial_cnt - len(df_merged)}건")

    df_merged['building_info_id'] = df_merged['building_info_id'].astype(int)
    df_pub['building_info_id'] = df_pub['building_info_id'].astype(int)
    df_pub = df_pub.sort_values('price_date')

    df_merged = pd.merge_asof(
        df_merged.sort_values('contract_date'), df_pub,
        left_on='contract_date', right_on='price_date',
        by='building_info_id', direction='backward'
    )

    if df_merged.empty:
        print("!!! 분석 대상 없음 !!!")
        return

    df_target = df_merged.copy()

    # (1) 시세 추정
    def estimate_market_price(row):
        if pd.notna(row['trade_price']) and row['trade_price'] > 0:
            return row['trade_price']
        if pd.notna(row['public_price']) and row['public_price'] > 0:
            m_use = str(row['main_use'])
            if any(x in m_use for x in ['다세대', '오피스텔', '연립', '근린']):
                return row['public_price'] * 1.8
            return row['public_price'] * 1.5
        return np.nan

    df_target['est_market_price'] = df_target.apply(estimate_market_price, axis=1)
    df_target = df_target.dropna(subset=['est_market_price'])
    df_target = df_target[df_target['est_market_price'] > 0]

    # (2) 전세가율 계산 (순수 전세 / 시세)
    df_target['jeonse_ratio'] = df_target['rent_price'] / df_target['est_market_price']
    df_target = df_target[df_target['jeonse_ratio'] < 3.0].copy()

    # (3) HUG 위험도
    filled_public = df_target['public_price'].fillna(df_target['est_market_price'] * 0.7)
    df_target['hug_safe_limit'] = (filled_public * 1.26).astype(int)
    df_target['hug_risk_ratio'] = df_target.apply(
        lambda x: x['rent_price'] / x['hug_safe_limit'] if x['hug_safe_limit'] > 0 else 0, axis=1
    )

    # (4) [수정됨] 정성적 리스크 점수화 (Simulation Score)
    # 가짜 대출금을 만드는 대신, 위험 '점수'를 계산합니다.

    # 4-1. 가중치 로직
    def type_weight(use):
        s = str(use)
        if '근린' in s: return 0.4
        if any(c in s for c in ['다세대', '오피스텔', '연립']): return 0.1
        return 0.0

    def trust_weight(owner):
        return 0.3 if owner and ('신탁' in str(owner)) else 0.0

    high_risk_regions = ['미추홀', '강서', '관악', '수원', '대전', '대구', '부평']

    def region_weight(row):
        addr = str(row.get('detail_address', '')) + str(row.get('lot_address', ''))
        return 0.2 if any(r in addr for r in high_risk_regions) else 0.0

    def short_term_weight(row):
        try:
            if pd.isna(row['contract_date']) or pd.isna(row['ownership_changed_date']): return 0.0
            days = (row['contract_date'] - row['ownership_changed_date']).days
            if -90 < days < 90: return 0.3
            if 0 <= days < 730: return 0.1
            return 0.0
        except:
            return 0.0

    # 4-2. 점수 합산 (0.0 ~ 1.0 범위)
    w_type = df_target['main_use'].apply(type_weight)
    w_trust = df_target['owner_name'].apply(trust_weight)
    w_region = df_target.apply(region_weight, axis=1)
    w_short = df_target.apply(short_term_weight, axis=1)

    # 이 점수는 'CAUTION' 등급을 매길 때 보조 지표로 사용
    df_target['risk_simulation_score'] = (w_type + w_trust + w_region + w_short).clip(0, 1.0)

    # (5) [수정됨] 깡통전세 위험도 (Financial Risk)
    # 실제 대출 정보(등기부 채권)가 없으므로 지금은 0으로 가정.
    # 추후 등기부 크롤링 데이터가 있으면 여기서 더해줍니다.
    real_debt_amount = 0

    # 공식: (전세금 + 실제대출) / 시세
    df_target['total_risk_ratio'] = (df_target['rent_price'] + real_debt_amount) / df_target['est_market_price']



    # 점수는 표시용으로 변환 (재무 위험 + 정성 위험 반영)
    df_target['risk_score'] = (
                (df_target['total_risk_ratio'] * 0.7 + df_target['risk_simulation_score'] * 0.3) * 100).astype(
        int).clip(0, 100)

    # (6) [수정됨] 등급 판정 로직

    def determine_risk_level(row):
        # 1. 재무적 위험 (Financial) - 즉시 위험
        if row['hug_risk_ratio'] > 1.0: return 'RISKY'  # 보증보험 불가
        if row['total_risk_ratio'] >= 0.8: return 'RISKY'  # 깡통전세 (80% 이상)

        if row['risk_score'] >= 80:
            return 'RISKY'

        if row['risk_score'] >= 60:
            return 'CAUTION'

        if row['risk_simulation_score'] >= 0.5: return 'CAUTION'

        return 'SAFE'

    df_target['risk_level'] = df_target.apply(determine_risk_level, axis=1)

    # DB 저장을 위해 loan_amount 컬럼은 0 또는 시뮬레이션 값으로 채워둠 (호환성 유지)
    # 단, total_risk_ratio 계산에는 이미 빠져있음.
    df_target['est_loan_amount'] = 0

    # --- AI 모델 적용 (Optional) ---
    model_path = os.path.join(project_root, 'models', 'fraud_rf_model.pkl')

    if os.path.exists(model_path):
        print("--- [AI] 학습된 모델로 예측 수행 ---")
        try:
            rf_model = joblib.load(model_path)

            # DataFrame의 각 행을 순회하며 입력 데이터 생성
            # (속도를 위해 벡터화 연산을 권장하지만, 로직 일관성을 위해 apply 사용)

            def get_ai_prob(row):
                # 1. 필요한 변수 준비 (data_processor와 동일 로직)
                short_term_w = 0.0
                if pd.notna(row['ownership_changed_date']) and pd.notna(row['contract_date']):
                    days = (row['contract_date'] - row['ownership_changed_date']).days
                    if days < 90:
                        short_term_w = 0.3
                    elif days < 730:
                        short_term_w = 0.1

                is_trust = 1 if row['owner_name'] and '신탁' in str(row['owner_name']) else 0
                is_viol = 1 if str(row['is_violating_building']).strip() == 'Y' else 0

                # 2. 피처 계산
                feats = calculate_risk_features(
                    deposit_amount=row['rent_price'],
                    market_price=row['est_market_price'],
                    real_debt=0,  # 배치 분석 시엔 등기부 빚 정보 없으므로 0
                    main_use=row['main_use'],
                    usage_approval_date=row['use_apr_day'],
                    is_illegal=is_viol,
                    is_trust_owner=is_trust,
                    short_term_weight=short_term_w
                )

                # 3. 모델 입력 포맷 변환 (DataFrame 1행)
                df_input = convert_to_model_input(feats)

                # 4. 예측
                return rf_model.predict_proba(df_input)[0][1]

            # 전체 데이터에 적용
            df_target['ai_risk_prob'] = df_target.apply(get_ai_prob, axis=1)
            print(f"-> AI 예측 완료 (평균 위험도: {df_target['ai_risk_prob'].mean():.4f})")

        except Exception as e:
            print(f"!!! [Critical] AI 모델 예측 실패: {e}")
            df_target['ai_risk_prob'] = 0.0

    # --- DB 저장 ---
    df_save = df_target[[
        'building_info_id', 'key', 'rent_price', 'est_market_price',
        'jeonse_ratio', 'hug_safe_limit', 'hug_risk_ratio',
        'total_risk_ratio', 'est_loan_amount', 'risk_level', 'risk_score', 'ai_risk_prob'
    ]].copy()

    df_save.rename(columns={
        'key': 'address_key',
        'rent_price': 'used_rent_price',
        'est_market_price': 'used_market_price',
        'est_loan_amount': 'estimated_loan_amount'
    }, inplace=True)

    df_save['created_at'] = datetime.now()
    df_save = df_save.sort_values('total_risk_ratio', ascending=False).drop_duplicates(subset=['address_key'])

    print(f"--- 최종 저장: {len(df_save)}건 ---")
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM risk_analysis_result"))
            conn.commit()
        df_save.to_sql('risk_analysis_result', engine, if_exists='append', index=False)
        print("-> 저장 완료")
    except Exception as e:
        print(f"저장 실패: {e}")


if __name__ == "__main__":
    run_risk_analysis_pipeline()