import pandas as pd
import numpy as np
import os
import sys
from sqlalchemy import text

# 프로젝트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from app.core.database import engine
from scripts.risk_pipeline import _create_join_key_robust, _generate_key_from_pnu

# 출력 옵션 설정 (데이터 다 보이게)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 20)


def run_debug_analysis():
    print("=" * 80)
    print("🕵️‍♂️ [데이터 정밀 진단] 매칭 오류 및 이상치 역추적 시작")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. 데이터 로드 (필요한 컬럼만 콕 집어서)
    # ---------------------------------------------------------
    print("\n>> 1. DB 데이터 샘플링 로드 중...")

    # (1) 전세 데이터 (건물명, 면적 포함)
    sql_rent = """
        SELECT 
            시군구, 법정동, 본번, 부번,
            보증금 as rent_price, 
            전용면적 as rent_area,
            건물명 as rent_bldg_name,  -- [확인용] 전세 계약서상 건물명
            건물유형 as rent_type,     -- [확인용] 아파트/단독다가구 등
            계약일 as contract_date
        FROM raw_rent
        WHERE 월세 = 0 AND 계약일 >= '20230101'
    """

    # (2) 건축물대장 (주용도, 면적 포함)
    sql_build = """
        SELECT 
            id as building_info_id, 
            unique_number, 
            main_use as build_main_use, -- [확인용] 건축물대장상 용도
            exclusive_area as build_area, -- [확인용] 건축물대장상 면적
            owner_name
        FROM building_info 
        WHERE unique_number IS NOT NULL
    """

    # (3) 공시지가
    sql_pub = "SELECT building_info_id, price as public_price FROM public_price_history"

    with engine.connect() as conn:
        df_rent = pd.read_sql(sql_rent, conn)
        df_build = pd.read_sql(sql_build, conn)
        df_pub = pd.read_sql(sql_pub, conn)

    # ---------------------------------------------------------
    # 2. 키 생성 및 매칭
    # ---------------------------------------------------------
    print(">> 2. 키 생성 및 병합 시도...")

    col_map = {'sgg': '시군구', 'bjd': '법정동', 'bon': '본번', 'bu': '부번'}
    df_rent['key'] = df_rent.apply(lambda row: _create_join_key_robust(row, col_map), axis=1)
    df_build['key'] = df_build['unique_number'].apply(_generate_key_from_pnu)

    # 병합 (Left Join)
    df_merged = pd.merge(df_rent, df_build, on='key', how='left')

    # 공시지가 병합
    df_merged = pd.merge(df_merged, df_pub, on='building_info_id', how='left')

    # 매칭 성공한 데이터만 분석
    df_matched = df_merged.dropna(subset=['building_info_id']).copy()
    print(f"   - 전세 데이터: {len(df_rent)}건")
    print(f"   - 건축물대장 매칭 성공: {len(df_matched)}건")

    # ---------------------------------------------------------
    # 3. [핵심] 이상치 정밀 분석 (Smoking Gun 찾기)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("🚨 [분석 1] '대참사' 매칭 케이스 (전세가율 200% 이상)")
    print("   -> 아파트 전세가 빌라 시세에 붙었는지 확인하세요.")
    print("=" * 80)

    # 타입 변환
    df_matched['rent_price'] = pd.to_numeric(df_matched['rent_price'], errors='coerce')
    df_matched['public_price'] = pd.to_numeric(df_matched['public_price'], errors='coerce') / 10000
    df_matched['rent_area'] = pd.to_numeric(df_matched['rent_area'], errors='coerce')
    df_matched['build_area'] = pd.to_numeric(df_matched['build_area'], errors='coerce')

    # 임시 전세가율 (공시지가 기준 1.5배 시세 가정)
    df_matched['temp_market_price'] = df_matched['public_price'] * 1.5
    df_matched['ratio'] = df_matched['rent_price'] / df_matched['temp_market_price']

    # 이상치 추출 (비율 2.0 이상)
    anomalies = df_matched[df_matched['ratio'] > 2.0].sort_values('ratio', ascending=False).head(10)

    if not anomalies.empty:
        print(anomalies[[
            'key',
            'rent_bldg_name', 'build_main_use',  # [비교 1] 이름 vs 용도
            'rent_area', 'build_area',  # [비교 2] 면적 vs 면적
            'rent_price', 'temp_market_price',  # [비교 3] 가격 vs 시세
            'ratio'
        ]])
    else:
        print("-> 이상치(200% 초과)가 발견되지 않았습니다. (좋은 징조!)")

    # ---------------------------------------------------------
    # 4. [핵심] 면적 불일치 분석
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("📏 [분석 2] 면적(Area) 차이 검증")
    print("   -> 같은 주소인데 평수가 다르면 '다른 집'입니다.")
    print("=" * 80)

    df_matched['area_diff'] = abs(df_matched['rent_area'] - df_matched['build_area'])

    # 10평(33m2) 이상 차이나는 경우
    area_mismatch = df_matched[df_matched['area_diff'] > 10.0].sort_values('area_diff', ascending=False).head(10)

    if not area_mismatch.empty:
        print(f"!!! 경고: 면적 차이가 큰 데이터가 {len(df_matched[df_matched['area_diff'] > 10.0])}건 발견됨 !!!")
        print(area_mismatch[[
            'key',
            'rent_bldg_name', 'rent_area',  # 전세 데이터
            'build_main_use', 'build_area',  # 건축물대장 데이터
            'area_diff'
        ]])
    else:
        print("-> 면적 차이가 큰 데이터가 없습니다.")

    # ---------------------------------------------------------
    # 5. [핵심] 건물 유형 불일치 분석
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("🏢 [분석 3] 건물 유형(Type) 불일치")
    print("   -> '아파트' 전세가 '단독주택/빌라' 대장에 붙었는지 확인")
    print("=" * 80)

    # 전세는 아파트인데, 대장은 아파트가 아닌 경우
    type_mismatch = df_matched[
        (df_matched['rent_type'] == '아파트') &
        (~df_matched['build_main_use'].str.contains('아파트', na=False))
        ].head(10)

    if not type_mismatch.empty:
        print(type_mismatch[[
            'key',
            'rent_bldg_name', 'rent_type',
            'build_main_use', 'owner_name'
        ]])
    else:
        print("-> 아파트-비아파트 혼동 매칭이 발견되지 않았습니다.")


if __name__ == "__main__":
    run_debug_analysis()

### 📊 이 코드를 실행하면 알 수 있는 것