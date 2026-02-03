import os
from sqlalchemy import text
from app.core.config import engine
# 좌표 캐시 파일 경로
COORD_CACHE_FILE = os.path.join(os.getcwd(), 'data', 'region_coords_cache.json')

# 1. 요약 데이터 조회 (최신 월 데이터만 WHERE 조건으로 필터링)
def fetch_latest_summaries(db):
    # 쿼리: 각 지역별로 가지고 있는 데이터 중 가장 최근 달(MAX month)의 데이터만 긁어옵니다.
    query = text("""
        SELECT 
            t.region_code,
            r.region_name,
            r.lat,
            r.lng,
            t.avg_ratio as latest_ratio,
            t.risk_level,
            t.tx_count,
            t.month as data_month 
        FROM (
            SELECT 
                *,
                ROW_NUMBER() OVER (PARTITION BY region_code ORDER BY month DESC) as rn
            FROM regional_stats
        ) t
        JOIN regions r ON t.region_code = r.region_code
        WHERE t.rn = 1; 
    """)

    rows = db.execute(query).mappings().fetchall()

    data_list = [dict(row) for row in rows]

    return {
        "count": len(data_list),
        "data": data_list
    }

# 2. 상세 히스토리 조회 (특정 지역 조건 필수)
def fetch_region_history(db, region_code, limit_months):
    # 1. 히스토리 데이터 조회 (ratio 별칭 사용으로 에러 해결)
    query = text("""
        SELECT month, avg_ratio as ratio, tx_count, risk_level 
        FROM regional_stats
        WHERE region_code = :code
        ORDER BY month DESC
        LIMIT :limit
    """)

    result = db.execute(query, {"code": region_code, "limit": limit_months})
    rows = result.mappings().fetchall()

    # 2. 데이터 변환 및 순서 뒤집기
    data_rows = [dict(r) for r in rows]
    data_rows.reverse()  # 과거 -> 최신 순으로 정렬

    # 3. 지역 이름 조회
    region_query = text("SELECT region_name, lat, lng FROM regions WHERE region_code = :c")

    region_row = db.execute(region_query, {"c": region_code}).mappings().fetchone()

    region_name = region_row['region_name']
    lat = region_row['lat']
    lng = region_row['lng']

    return {
        "region_code": region_code,
        "region_name": region_name,
        "lat": lat,
        "lng": lng,
        "history": data_rows
    }