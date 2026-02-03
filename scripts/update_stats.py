import sys
import os
import pandas as pd
from sqlalchemy import text

from app.core.database import engine


def categorize_building_type(raw_type):
    """건물유형 분류"""
    if pd.isna(raw_type):
        return '기타'
    raw_type = str(raw_type).strip()

    if '아파트' in raw_type:
        return '아파트'
    if '오피스텔' in raw_type:
        return '오피스텔'
    if any(x in raw_type for x in ['연립', '다세대', '빌라', '단독', '다가구']):
        return '연립다세대'
    return '기타'


def update_regional_stats():
    print("🚀 [Batch] 지역별 전세가율 통계 집계 시작...")

    # ---------------------------------------------------------
    # 1. DB에서 원천 데이터 가져오기 (Extract)
    # ---------------------------------------------------------
    print("   ㄴ 1. 데이터 조회 중 (raw_rent, raw_trade)...")

    # (1) 전세 데이터 (월세가 0인 것만) - 건물유형 포함
    sql_rent = """
        SELECT 시군구, 법정동, 본번, 부번, 보증금, 계약일, 건물명, 건물유형
        FROM raw_rent 
        WHERE 월세 = '0' OR 월세 IS NULL
    """

    # (2) 매매 데이터
    sql_trade = """
        SELECT 시군구, 법정동, 본번, 부번, 거래금액, 계약일
        FROM raw_trade 
    """

    with engine.connect() as conn:
        df_rent = pd.read_sql(text(sql_rent), conn)
        df_trade = pd.read_sql(text(sql_trade), conn)

    if df_rent.empty or df_trade.empty:
        print("⚠️ 데이터가 부족하여 집계를 중단합니다.")
        return

    # ---------------------------------------------------------
    # 2. 데이터 전처리 (Transform)
    # ---------------------------------------------------------
    print("   ㄴ 2. 데이터 전처리 및 병합 중...")

    # (1) 금액 컬럼 숫자 변환 (콤마 제거)
    df_rent['deposit'] = pd.to_numeric(df_rent['보증금'].str.replace(',', ''), errors='coerce')
    df_trade['price'] = pd.to_numeric(df_trade['거래금액'].str.replace(',', ''), errors='coerce')

    # (2) 날짜 변환 (YYYYMMDD -> datetime)
    df_rent['date'] = pd.to_datetime(df_rent['계약일'], format='%Y%m%d', errors='coerce')
    df_trade['date'] = pd.to_datetime(df_trade['계약일'], format='%Y%m%d', errors='coerce')

    # (3) 건물유형 분류
    df_rent['building_type'] = df_rent['건물유형'].apply(categorize_building_type)

    # (4) 고유 키 생성 (시군구+법정동+본번+부번)
    def make_key(row):
        return f"{str(row['시군구'])}-{str(row['법정동'])}-{str(row['본번'])}-{str(row['부번'])}"

    df_rent['key'] = df_rent.apply(make_key, axis=1)
    df_trade['key'] = df_trade.apply(make_key, axis=1)

    # (5) 정렬 (merge_asof를 위해 날짜순 정렬 필수)
    df_rent = df_rent.sort_values('date')
    df_trade = df_trade.sort_values('date')

    # (6) 매매가 매칭 (merge_asof)
    df_merged = pd.merge_asof(
        df_rent,
        df_trade[['key', 'price', 'date']],
        on='date',
        by='key',
        direction='backward',
        tolerance=pd.Timedelta(days=365 * 2)
    )

    # 매매가 없는 데이터 제거
    df_final = df_merged.dropna(subset=['price'])

    # (7) 전세가율 계산
    df_final['ratio'] = (df_final['deposit'] / df_final['price']) * 100

    # 이상치 제거 (전세가율 200% 이상)
    df_final = df_final[df_final['ratio'] <= 200]

    # (8) 월별 문자열 생성
    df_final['month'] = df_final['date'].dt.strftime('%Y-%m')

    # ---------------------------------------------------------
    # 3. 집계 (전체 + 건물유형별)
    # ---------------------------------------------------------
    print("   ㄴ 3. 통계 집계 중...")

    # (A) 전체 통계 (기존 방식)
    stats_all = df_final.groupby(['시군구', 'month']).agg(
        avg_ratio=('ratio', 'mean'),
        tx_count=('ratio', 'count')
    ).reset_index()
    stats_all['building_type'] = 'ALL'  # 전체 표시

    # (B) 건물유형별 통계 (신규)
    stats_by_type = df_final.groupby(['시군구', 'month', 'building_type']).agg(
        avg_ratio=('ratio', 'mean'),
        tx_count=('ratio', 'count')
    ).reset_index()

    # 합치기
    stats = pd.concat([stats_all, stats_by_type], ignore_index=True)

    # (9) 위험 등급 산정
    def get_risk_level(r):
        if r >= 80:
            return 'RISKY'
        elif r >= 70:
            return 'CAUTION'
        return 'SAFE'

    stats['risk_level'] = stats['avg_ratio'].apply(get_risk_level)

    # 컬럼명 DB 포맷에 맞게 변경
    stats.rename(columns={
        '시군구': 'region_code',
    }, inplace=True)

    # 지역명은 임시로 코드 사용
    stats['region_name'] = stats['region_code']

    print(f"   ㄴ 집계 완료: 총 {len(stats)}건의 통계 데이터 생성")
    print(f"      - 전체(ALL): {len(stats[stats['building_type'] == 'ALL'])}건")
    print(f"      - 아파트: {len(stats[stats['building_type'] == '아파트'])}건")
    print(f"      - 연립다세대: {len(stats[stats['building_type'] == '연립다세대'])}건")
    print(f"      - 오피스텔: {len(stats[stats['building_type'] == '오피스텔'])}건")

    # ---------------------------------------------------------
    # 4. DB 적재 (Load)
    # ---------------------------------------------------------
    print("   ㄴ 4. DB에 저장 중 (regional_stats)...")

    with engine.begin() as conn:
        # 기존 데이터 삭제 (Full Refresh 전략)
        conn.execute(text("TRUNCATE TABLE regional_stats"))

        # 데이터프레임 -> DB Insert
        stats.to_sql('regional_stats', con=conn, if_exists='append', index=False)

    print("✅ [Success] 통계 데이터 갱신 완료!")


if __name__ == "__main__":
    update_regional_stats()