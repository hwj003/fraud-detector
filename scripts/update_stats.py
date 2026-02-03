import sys
import os
import pandas as pd
from sqlalchemy import text

from scripts.db_manager import engine

def update_regional_stats():
    print("🚀 [Batch] 지역별 전세가율 통계 집계 시작...")

    # ---------------------------------------------------------
    # 1. DB에서 원천 데이터 가져오기 (Extract)
    # ---------------------------------------------------------
    print("   ㄴ 1. 데이터 조회 중 (raw_rent, raw_trade)...")

    # (1) 전세 데이터 (월세가 0인 것만, 최근 1년치 예시)
    sql_rent = """
        SELECT 시군구, 법정동, 본번, 부번, 보증금, 계약일, 건물명
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
    # MySQL에서 VARCHAR로 저장되어 있으므로 변환 필수
    df_rent['deposit'] = pd.to_numeric(df_rent['보증금'].str.replace(',', ''), errors='coerce')
    df_trade['price'] = pd.to_numeric(df_trade['거래금액'].str.replace(',', ''), errors='coerce')

    # (2) 날짜 변환 (YYYYMMDD -> datetime)
    df_rent['date'] = pd.to_datetime(df_rent['계약일'], format='%Y%m%d', errors='coerce')
    df_trade['date'] = pd.to_datetime(df_trade['계약일'], format='%Y%m%d', errors='coerce')

    # (3) 고유 키 생성 (시군구+법정동+본번+부번) -> 같은 건물을 찾기 위함
    # 예: 11110 + 10100 + 0001 + 0002
    def make_key(row):
        return f"{str(row['시군구'])}-{str(row['법정동'])}-{str(row['본번'])}-{str(row['부번'])}"

    df_rent['key'] = df_rent.apply(make_key, axis=1)
    df_trade['key'] = df_trade.apply(make_key, axis=1)

    # (4) 정렬 (merge_asof를 위해 날짜순 정렬 필수)
    df_rent = df_rent.sort_values('date')
    df_trade = df_trade.sort_values('date')

    # (5) 매매가 매칭 (merge_asof)
    # 로직: 전세 계약일 이전에 발생한, 같은 건물의 가장 최근 매매가를 가져옴
    df_merged = pd.merge_asof(
        df_rent,
        df_trade[['key', 'price', 'date']],  # 필요한 컬럼만
        on='date',  # 기준 시간
        by='key',  # 기준 ID (같은 건물)
        direction='backward',  # 과거 데이터 탐색
        tolerance=pd.Timedelta(days=365 * 2)  # 최대 2년 전 매매가까지만 인정
    )

    # 매매가 없는 데이터(신축이라 매매 기록 없음 등)는 계산 불가 -> 제거
    df_final = df_merged.dropna(subset=['price'])

    # (6) 전세가율 계산
    df_final['ratio'] = (df_final['deposit'] / df_final['price']) * 100

    # 이상치 제거 (전세가율 200% 이상은 오기입일 확률 높음)
    df_final = df_final[df_final['ratio'] <= 200]

    # (7) 집계 (Group By)
    # 월별 문자열 생성 (YYYY-MM)
    df_final['month'] = df_final['date'].dt.strftime('%Y-%m')

    # 시군구명 매핑 (데이터에 시군구명이 없으므로 코드로 그룹화)
    # 실제로는 meta 테이블과 조인해서 이름을 가져와야 하지만, 여기선 코드만 사용하거나 raw 데이터에 이름이 있다면 사용
    # 여기서는 편의상 region_code만 저장

    stats = df_final.groupby(['시군구', 'month']).agg(
        avg_ratio=('ratio', 'mean'),
        tx_count=('ratio', 'count')
    ).reset_index()

    # (8) 위험 등급 산정
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
        'month': 'month',
        'avg_ratio': 'avg_ratio',
        'tx_count': 'tx_count',
        'risk_level': 'risk_level'
    }, inplace=True)

    # 지역명(region_name)은 API 조회 시 join하거나, 여기서 매핑 테이블을 써야 함.
    # 임시로 region_code를 name으로 넣음 (추후 개선 포인트)
    stats['region_name'] = stats['region_code']

    print(f"   ㄴ 집계 완료: 총 {len(stats)}건의 통계 데이터 생성")

    # ---------------------------------------------------------
    # 3. DB 적재 (Load)
    # ---------------------------------------------------------
    print("   ㄴ 3. DB에 저장 중 (regional_stats)...")

    with engine.begin() as conn:
        # 기존 데이터 삭제 (Full Refresh 전략) - 중복 에러 방지
        conn.execute(text("TRUNCATE TABLE regional_stats"))

        # 데이터프레임 -> DB Insert
        stats.to_sql('regional_stats', con=conn, if_exists='append', index=False)

    print("✅ [Success] 통계 데이터 갱신 완료!")

if __name__ == "__main__":
    update_regional_stats()