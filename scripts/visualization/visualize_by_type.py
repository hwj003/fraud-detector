import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import platform
from sqlalchemy import text

# ---------------------------------------------------------
# 1. 프로젝트 설정
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from app.core.database import engine
from scripts.data_processor import _create_join_key_from_columns


# ---------------------------------------------------------
# 2. 한글 폰트 설정
# ---------------------------------------------------------
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    elif system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    else:
        plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False)


# ---------------------------------------------------------
# [NEW] 3. DB 메타 테이블을 활용한 지역명 매핑 함수
# ---------------------------------------------------------
def get_sigungu_map_from_db():
    """
    meta_bjdong_codes 테이블을 조회하여
    {'11110': '서울 종로구', ...} 형태의 매핑 딕셔너리를 생성합니다.
    """
    print(">> [Meta] 지역 코드 매핑 테이블 조회 중...")

    # 중복 제거를 위해 DISTINCT 사용 (같은 구에 여러 동이 있으므로)
    sql = """
        SELECT DISTINCT sgg_code, bjdong_name 
        FROM meta_bjdong_codes
    """

    try:
        with engine.connect() as conn:
            df_meta = pd.read_sql(sql, conn)

        mapping = {}
        # 파싱 로직: "서울특별시 종로구 사직동" -> "서울 종로구"
        for _, row in df_meta.iterrows():
            code = str(row['sgg_code']).strip()
            full_name = str(row['bjdong_name']).strip()

            # 이미 매핑된 코드는 건너뜀 (성능 최적화)
            if code in mapping:
                continue

            tokens = full_name.split()
            if len(tokens) >= 2:
                sido = tokens[0]  # 서울특별시
                gugun = tokens[1]  # 종로구

                # 시도 명칭 줄이기 (그래프 공간 확보용)
                if '특별' in sido or '광역' in sido:
                    sido = sido[:2]  # 서울특별시 -> 서울
                elif '경상' in sido or '전라' in sido or '충청' in sido:
                    sido = sido[0] + sido[2]  # 경상남도 -> 경남
                elif '경기' in sido:
                    sido = '경기'
                elif '강원' in sido:
                    sido = '강원'
                elif '제주' in sido:
                    sido = '제주'

                short_name = f"{sido} {gugun}"
                mapping[code] = short_name
            else:
                # 예외 케이스: 세종특별자치시 등
                mapping[code] = tokens[0]

        print(f"   -> {len(mapping)}개의 지역 코드 매핑 로드 완료")
        return mapping

    except Exception as e:
        print(f"⚠️ [Warning] 메타 테이블 조회 실패: {e}")
        print("   -> 코드 그대로 출력됩니다.")
        return {}


# ---------------------------------------------------------
# 4. 데이터 로드 및 전처리 (Pure Logic)
# ---------------------------------------------------------
def load_pure_market_data():
    print(">> 1. 실거래가(매매/전세) 데이터 로드 중...")

    sql_rent = """
        SELECT 시군구, 법정동, 본번, 부번, 
               보증금 as rent_price, 계약일 as contract_date, 
               전용면적 as area, 건물유형 as building_type
        FROM raw_rent
        WHERE 월세 = 0 AND 계약일 >= '20230101'
    """

    sql_trade = """
        SELECT 시군구, 법정동, 본번, 부번, 
               거래금액 as trade_price, 계약일 as trade_date
        FROM raw_trade
        WHERE 계약일 >= '20230101'
    """

    with engine.connect() as conn:
        df_rent = pd.read_sql(sql_rent, conn)
        df_trade = pd.read_sql(sql_trade, conn)

    for df in [df_rent, df_trade]:
        df['key'] = df.apply(lambda row: _create_join_key_from_columns(row), axis=1)

    df_rent['rent_price'] = pd.to_numeric(df_rent['rent_price'], errors='coerce')
    df_trade['trade_price'] = pd.to_numeric(df_trade['trade_price'], errors='coerce')
    df_rent['contract_date'] = pd.to_datetime(df_rent['contract_date'])
    df_trade['trade_date'] = pd.to_datetime(df_trade['trade_date'])

    df_rent = df_rent.sort_values('contract_date')
    df_trade = df_trade.sort_values('trade_date')

    df_merged = pd.merge_asof(
        df_rent, df_trade[['key', 'trade_price', 'trade_date']],
        left_on='contract_date', right_on='trade_date',
        by='key', direction='backward', tolerance=pd.Timedelta(days=365 * 2)
    )

    df_clean = df_merged.dropna(subset=['trade_price']).copy()
    df_clean['jeonse_ratio'] = (df_clean['rent_price'] / df_clean['trade_price']) * 100
    df_clean = df_clean[df_clean['jeonse_ratio'] <= 200]

    return df_clean


# ---------------------------------------------------------
# 5. 건물 유형 분류 함수
# ---------------------------------------------------------
def categorize_building_type(raw_type):
    if pd.isna(raw_type): return '기타'
    raw_type = str(raw_type).strip()

    if '아파트' in raw_type: return '아파트'
    if '오피스텔' in raw_type: return '오피스텔'
    if any(x in raw_type for x in ['연립', '다세대', '빌라', '단독', '다가구']):
        return '연립/다세대'
    return '기타'


# ---------------------------------------------------------
# 6. 시각화 실행
# ---------------------------------------------------------
def run_type_visualization():
    # 1. 데이터 로드
    df = load_pure_market_data()

    # 2. [NEW] DB에서 지역명 매핑 가져오기
    sigungu_map = get_sigungu_map_from_db()

    # 매핑 적용 (매핑 없으면 원래 코드 사용)
    df['region_name'] = df['시군구'].astype(str).map(sigungu_map).fillna(df['시군구'])

    # 3. 유형 분류 및 필터링
    df['category'] = df['building_type'].apply(categorize_building_type)
    target_types = ['아파트', '오피스텔', '연립/다세대']
    df_target = df[df['category'].isin(target_types)].copy()

    if df_target.empty:
        print("❌ 시각화할 데이터가 없습니다.")
        return

    print(f"\n>> 분석 데이터 건수: {len(df_target)}건")

    # 4. 시각화
    set_korean_font()
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=False)

    palette_map = {'아파트': 'Blues_r', '오피스텔': 'Oranges_r', '연립/다세대': 'Reds_r'}

    for idx, b_type in enumerate(target_types):
        ax = axes[idx]
        subset = df_target[df_target['category'] == b_type]

        # 지역별 평균 계산
        group = subset.groupby('region_name')['jeonse_ratio'].agg(['mean', 'count']).reset_index()
        group = group[group['count'] >= 10]
        group = group.sort_values(by='mean', ascending=False).head(15)

        sns.barplot(
            data=group, x='mean', y='region_name',
            ax=ax, palette=palette_map[b_type]
        )

        ax.axvline(x=80, color='red', linestyle='--', linewidth=1.5, label='위험선(80%)')
        ax.set_title(f"{b_type} 전세가율 Top 15", fontsize=14, fontweight='bold')
        ax.set_xlabel('전세가율(%)')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        for i, v in enumerate(group['mean']):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)

    plt.suptitle("건물 유형별 평균 전세가율 비교 (Pure Data)", fontsize=16, y=0.98)
    plt.tight_layout()

    save_path = os.path.join(project_root, 'models', 'jeonse_ratio_by_type.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프 저장 완료: {save_path}")


if __name__ == "__main__":
    run_type_visualization()