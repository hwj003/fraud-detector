from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
# 위에서 정의한 schema 임포트
from app.schemas import ResponseEnvelope, Meta
from scripts.db_manager import get_db
from app.services.map_service import fetch_latest_summaries, fetch_region_history
router = APIRouter(prefix="/stats", tags=["Statistics"])

# ------------------------------------------------------------------
# 1. [지도용] 요약 API (가볍고 빠름)
# GET /stats/summary
# ------------------------------------------------------------------
@router.get("/summary", response_model=ResponseEnvelope)
def get_region_summaries(db: Session = Depends(get_db)):
    """
    전국 모든 구의 '최신' 데이터만 가져옵니다. (지도 마커용)
    """
    # Service 계층 호출
    summaries = fetch_latest_summaries(db)

    return ResponseEnvelope(
        meta=Meta(code=200, message="지역별 요약 조회 성공"),
        data=summaries # List[RegionSummary]
    )

# ------------------------------------------------------------------
# 2. [차트용] 상세 API (특정 지역만 조회)
# GET /stats/history?region_code=11110
# ------------------------------------------------------------------
@router.get("/history", response_model=ResponseEnvelope)
def get_region_history (
        region_code: str = Query(..., description="법정동 코드 (예: 11110)"),
        months: int = Query(12, description="최근 N개월 데이터 조회"),
        db: Session = Depends(get_db)
):
    """
    특정 구의 '과거' 히스토리를 가져옵니다.
    """
    # Service 계층 호출
    history_data = fetch_region_history(db, region_code, months)

    if not history_data:
        return ResponseEnvelope(
            meta=Meta(code=404, message="해당 지역의 데이터가 없습니다."),
            data=None
        )

    return ResponseEnvelope(
        meta=Meta(code=200, message="상세 히스토리 조회 성공"),
        data=history_data
    )