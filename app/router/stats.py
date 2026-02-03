"""
Stats API Router - 건물유형별 전세가율 통계

엔드포인트:
- GET /stats/summary - 지역별 전세가율 요약 (건물유형 필터 지원)
- GET /stats/history/{region_code} - 특정 지역 히스토리 (건물유형 필터 지원)
- GET /stats/building-types - 건물유형별 전체 요약
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional, Literal
from app.core.database import get_db

router = APIRouter(prefix="/stats", tags=["Statistics"])

# 허용되는 건물유형
BuildingType = Literal["ALL", "아파트", "연립다세대", "오피스텔"]


@router.get("/summary", summary="지역별 전세가율 요약")
def get_regional_summary(
        building_type: Optional[str] = Query(
            default="ALL",
            description="건물유형 필터 (ALL, 아파트, 연립다세대, 오피스텔)"
        ),
        db: Session = Depends(get_db)
):
    """
    지역별 최신 전세가율 요약 데이터를 반환합니다.

    **Query Parameters:**
    - `building_type`: 건물유형 필터 (기본값: ALL)
        - `ALL`: 전체 통계
        - `아파트`: 아파트만
        - `연립다세대`: 연립/다세대/빌라
        - `오피스텔`: 오피스텔만

    **Response:**
    ```json
    {
        "meta": {"code": 200, "message": "success"},
        "data": {
            "building_type": "아파트",
            "count": 50,
            "regions": [
                {
                    "region_code": "11110",
                    "region_name": "서울 종로구",
                    "lat": 37.5735,
                    "lng": 126.9788,
                    "latest_ratio": 75.3,
                    "risk_level": "CAUTION",
                    "tx_count": 234,
                    "data_month": "2025-01"
                }
            ]
        }
    }
    ```
    """
    # 건물유형 검증
    valid_types = ["ALL", "아파트", "연립다세대", "오피스텔"]
    if building_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 건물유형입니다. 허용값: {valid_types}"
        )

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
        "meta": {"code": 200, "message": "success"},
        "data": {
            "building_type": building_type,
            "count": len(data_list),
            "regions": data_list
        }
    }


@router.get("/history/{region_code}", summary="지역별 전세가율 히스토리")
def get_region_history(
        region_code: str,
        building_type: Optional[str] = Query(
            default="ALL",
            description="건물유형 필터 (ALL, 아파트, 연립다세대, 오피스텔)"
        ),
        months: int = Query(default=12, ge=1, le=36, description="조회할 개월 수"),
        db: Session = Depends(get_db)
):
    """
    특정 지역의 월별 전세가율 히스토리를 반환합니다.

    **Path Parameters:**
    - `region_code`: 시군구 코드 (예: 11110)

    **Query Parameters:**
    - `building_type`: 건물유형 필터 (기본값: ALL)
    - `months`: 조회할 개월 수 (기본값: 12, 최대: 36)

    **Response:**
    ```json
    {
        "meta": {"code": 200, "message": "success"},
        "data": {
            "region_code": "11110",
            "region_name": "서울 종로구",
            "building_type": "아파트",
            "lat": 37.5735,
            "lng": 126.9788,
            "history": [
                {"month": "2024-02", "ratio": 72.5, "tx_count": 45, "risk_level": "CAUTION"},
                {"month": "2024-03", "ratio": 73.1, "tx_count": 52, "risk_level": "CAUTION"}
            ]
        }
    }
    ```
    """
    # 건물유형 검증
    valid_types = ["ALL", "아파트", "연립다세대", "오피스텔"]
    if building_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 건물유형입니다. 허용값: {valid_types}"
        )

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
        "limit": months
    })
    rows = result.mappings().fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"해당 지역({region_code}) 또는 건물유형({building_type})의 데이터가 없습니다"
        )

    # 데이터 변환 및 순서 뒤집기 (과거 -> 최신)
    data_rows = [dict(r) for r in rows]
    data_rows.reverse()

    # 지역 정보 조회
    region_query = text("SELECT region_name, lat, lng FROM regions WHERE region_code = :c")
    region_row = db.execute(region_query, {"c": region_code}).mappings().fetchone()

    if not region_row:
        raise HTTPException(
            status_code=404,
            detail=f"지역 코드({region_code})를 찾을 수 없습니다"
        )

    return {
        "meta": {"code": 200, "message": "success"},
        "data": {
            "region_code": region_code,
            "region_name": region_row['region_name'],
            "building_type": building_type,
            "lat": region_row['lat'],
            "lng": region_row['lng'],
            "history": data_rows
        }
    }


@router.get("/building-types", summary="건물유형별 전체 요약")
def get_building_type_summary(db: Session = Depends(get_db)):
    """
    건물유형별 전체 평균 전세가율 요약을 반환합니다.

    **Response:**
    ```json
    {
        "meta": {"code": 200, "message": "success"},
        "data": {
            "아파트": {
                "avg_ratio": 68.5,
                "total_tx_count": 12345,
                "risk_level": "SAFE",
                "region_count": 150
            },
            "연립다세대": {
                "avg_ratio": 82.3,
                "total_tx_count": 8765,
                "risk_level": "RISKY",
                "region_count": 145
            },
            "오피스텔": {
                "avg_ratio": 75.1,
                "total_tx_count": 5432,
                "risk_level": "CAUTION",
                "region_count": 120
            }
        }
    }
    ```
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

    return {
        "meta": {"code": 200, "message": "success"},
        "data": result
    }


@router.get("/compare/{region_code}", summary="지역 내 건물유형별 비교")
def compare_building_types(
        region_code: str,
        db: Session = Depends(get_db)
):
    """
    특정 지역의 건물유형별 전세가율을 비교합니다.

    **Path Parameters:**
    - `region_code`: 시군구 코드 (예: 11110)

    **Response:**
    ```json
    {
        "meta": {"code": 200, "message": "success"},
        "data": {
            "region_code": "11110",
            "region_name": "서울 종로구",
            "comparison": {
                "아파트": {"ratio": 65.2, "tx_count": 123, "risk_level": "SAFE"},
                "연립다세대": {"ratio": 78.5, "tx_count": 87, "risk_level": "CAUTION"},
                "오피스텔": {"ratio": 82.1, "tx_count": 45, "risk_level": "RISKY"}
            }
        }
    }
    ```
    """
    # 지역 정보 조회
    region_query = text("SELECT region_name FROM regions WHERE region_code = :c")
    region_row = db.execute(region_query, {"c": region_code}).mappings().fetchone()

    if not region_row:
        raise HTTPException(
            status_code=404,
            detail=f"지역 코드({region_code})를 찾을 수 없습니다"
        )

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
        "meta": {"code": 200, "message": "success"},
        "data": {
            "region_code": region_code,
            "region_name": region_row['region_name'],
            "comparison": comparison
        }
    }