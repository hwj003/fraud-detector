"""
Map Service - 건물유형별 전세가율 통계 조회

지도 시각화를 위한 데이터 조회 서비스입니다.
건물유형(아파트/연립다세대/오피스텔)별 필터링을 지원합니다.
"""
import os
from sqlalchemy import text
from app.core.database import engine

# 좌표 캐시 파일 경로
COORD_CACHE_FILE = os.path.join(os.getcwd(), 'data', 'region_coords_cache.json')

# 허용되는 건물유형
VALID_BUILDING_TYPES = ["ALL", "아파트", "연립다세대", "오피스텔"]


def fetch_latest_summaries(db, building_type: str = "ALL"):
    """
    지역별 최신 전세가율 요약 데이터 조회

    Args:
        db: 데이터베이스 세션
        building_type: 건물유형 필터 (ALL, 아파트, 연립다세대, 오피스텔)

    Returns:
        dict: {
            "building_type": "아파트",
            "count": 50,
            "data": [...]
        }
    """
    # 건물유형 검증
    if building_type not in VALID_BUILDING_TYPES:
        building_type = "ALL"

    query = text("""
        SELECT 
            t.region_code,
            r.region_name,
            r.lat,
            r.lng,
            t.avg_ratio as latest_ratio,
            t.risk_level,
            t.tx_count,
            t.month as data_month,
            t.building_type
        FROM (
            SELECT 
                *,
                ROW_NUMBER() OVER (PARTITION BY region_code ORDER BY month DESC) as rn
            FROM regional_stats
            WHERE building_type = :building_type
        ) t
        JOIN regions r ON t.region_code = r.region_code
        WHERE t.rn = 1
        ORDER BY t.avg_ratio DESC
    """)

    rows = db.execute(query, {"building_type": building_type}).mappings().fetchall()
    data_list = [dict(row) for row in rows]

    return {
        "building_type": building_type,
        "count": len(data_list),
        "data": data_list
    }


def fetch_region_history(db, region_code: str, limit_months: int = 12, building_type: str = "ALL"):
    """
    특정 지역의 월별 전세가율 히스토리 조회

    Args:
        db: 데이터베이스 세션
        region_code: 시군구 코드
        limit_months: 조회할 개월 수
        building_type: 건물유형 필터

    Returns:
        dict: {
            "region_code": "11110",
            "region_name": "서울 종로구",
            "building_type": "아파트",
            "lat": 37.5735,
            "lng": 126.9788,
            "history": [...]
        }
    """
    # 건물유형 검증
    if building_type not in VALID_BUILDING_TYPES:
        building_type = "ALL"

    # 히스토리 데이터 조회
    query = text("""
        SELECT month, avg_ratio as ratio, tx_count, risk_level 
        FROM regional_stats
        WHERE region_code = :code AND building_type = :building_type
        ORDER BY month DESC
        LIMIT :limit
    """)

    result = db.execute(query, {
        "code": region_code,
        "building_type": building_type,
        "limit": limit_months
    })
    rows = result.mappings().fetchall()

    # 데이터 변환 및 순서 뒤집기 (과거 -> 최신)
    data_rows = [dict(r) for r in rows]
    data_rows.reverse()

    # 지역 정보 조회
    region_query = text("SELECT region_name, lat, lng FROM regions WHERE region_code = :c")
    region_row = db.execute(region_query, {"c": region_code}).mappings().fetchone()

    if not region_row:
        return None

    return {
        "region_code": region_code,
        "region_name": region_row['region_name'],
        "building_type": building_type,
        "lat": region_row['lat'],
        "lng": region_row['lng'],
        "history": data_rows
    }


def fetch_building_type_summary(db):
    """
    건물유형별 전체 요약 통계 조회

    Returns:
        dict: {
            "아파트": {"avg_ratio": 68.5, "total_tx_count": 12345, ...},
            "연립다세대": {...},
            "오피스텔": {...}
        }
    """
    query = text("""
        SELECT 
            building_type,
            ROUND(AVG(avg_ratio), 1) as avg_ratio,
            SUM(tx_count) as total_tx_count,
            COUNT(DISTINCT region_code) as region_count
        FROM regional_stats
        WHERE building_type != 'ALL' AND building_type != '기타'
        GROUP BY building_type
    """)

    rows = db.execute(query).mappings().fetchall()

    def get_risk_level(ratio):
        if ratio >= 80:
            return 'RISKY'
        elif ratio >= 70:
            return 'CAUTION'
        return 'SAFE'

    result = {}
    for row in rows:
        result[row['building_type']] = {
            "avg_ratio": float(row['avg_ratio']),
            "total_tx_count": int(row['total_tx_count']),
            "risk_level": get_risk_level(row['avg_ratio']),
            "region_count": int(row['region_count'])
        }

    return result


def fetch_region_comparison(db, region_code: str):
    """
    특정 지역의 건물유형별 전세가율 비교

    Args:
        db: 데이터베이스 세션
        region_code: 시군구 코드

    Returns:
        dict: {
            "region_code": "11110",
            "region_name": "서울 종로구",
            "comparison": {
                "아파트": {"ratio": 65.2, ...},
                "연립다세대": {...},
                "오피스텔": {...}
            }
        }
    """
    # 지역 정보 조회
    region_query = text("SELECT region_name FROM regions WHERE region_code = :c")
    region_row = db.execute(region_query, {"c": region_code}).mappings().fetchone()

    if not region_row:
        return None

    # 건물유형별 최신 데이터 조회
    query = text("""
        SELECT 
            building_type,
            avg_ratio as ratio,
            tx_count,
            risk_level,
            month
        FROM (
            SELECT 
                *,
                ROW_NUMBER() OVER (PARTITION BY building_type ORDER BY month DESC) as rn
            FROM regional_stats
            WHERE region_code = :code 
              AND building_type IN ('아파트', '연립다세대', '오피스텔')
        ) t
        WHERE rn = 1
    """)

    rows = db.execute(query, {"code": region_code}).mappings().fetchall()

    comparison = {}
    for row in rows:
        comparison[row['building_type']] = {
            "ratio": float(row['ratio']),
            "tx_count": int(row['tx_count']),
            "risk_level": row['risk_level'],
            "data_month": row['month']
        }

    return {
        "region_code": region_code,
        "region_name": region_row['region_name'],
        "comparison": comparison
    }