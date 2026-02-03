# scripts/update_coords.py
import json
import os
import sys
from sqlalchemy import text

# 프로젝트 루트 경로 설정 (db_manager 찾기 위해)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from app.core.database import engine

# 좌표 캐시 파일 위치 (본인 프로젝트 구조에 맞게 수정 확인)
JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'region_coords_cache.json')


def update_region_coordinates():
    print("🚀 지역 좌표 데이터 DB 업데이트 시작...")

    # 1. JSON 파일 읽기
    if not os.path.exists(JSON_PATH):
        print(f"❌ 파일이 없습니다: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        coords_data = json.load(f)

    print(f"   ㄴ 캐시 파일 로드 완료 ({len(coords_data)}개 지역)")

    # 2. DB 업데이트
    updated_count = 0

    with engine.begin() as conn:
        for region_code, coords in coords_data.items():
            # coords는 [lat, lng] 형태의 리스트라고 가정
            if not coords or len(coords) < 2:
                continue

            lat, lng = coords[0], coords[1]

            # 0.0 인 좌표는 업데이트 스킵 (유효하지 않음)
            if lat == 0 and lng == 0:
                continue

            # UPDATE 쿼리 실행
            stmt = text("""
                UPDATE regions 
                SET lat = :lat, lng = :lng 
                WHERE region_code = :code
            """)

            result = conn.execute(stmt, {"lat": lat, "lng": lng, "code": region_code})

            if result.rowcount > 0:
                updated_count += 1

    print(f"✅ 업데이트 완료: 총 {updated_count}개 지역의 좌표가 저장되었습니다.")


if __name__ == "__main__":
    update_region_coordinates()