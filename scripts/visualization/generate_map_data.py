import json
import requests
import os
import sys
import time

# ---------------------------------------------------------
# 1. 프로젝트 설정 & 모듈 로드
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# 기존 로직 재사용
from scripts.visualization.visualize_by_type import load_pure_market_data, get_sigungu_map_from_db

# [필수] Kakao REST API 키 입력 (또는 os.getenv('KAKAO_API_KEY') 사용)
KAKAO_API_KEY = os.getenv('KAKAO_API_KEY')

# 좌표 데이터를 저장할 캐시 파일 (API 호출 최소화용)
COORD_CACHE_FILE = os.path.join(project_root, 'data', 'region_coords_cache.json')


# ---------------------------------------------------------
# 2. Kakao API 연동 함수
# ---------------------------------------------------------
def get_coordinates_from_kakao(address):
    """
    Kakao Local API를 사용하여 주소의 좌표(x, y)를 반환합니다.
    """
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        result = response.json()

        if result['documents']:
            # 가장 첫 번째 검색 결과 사용
            # x: 경도(lng), y: 위도(lat)
            x = result['documents'][0]['x']
            y = result['documents'][0]['y']
            return float(y), float(x)
        else:
            print(f"   [API] 검색 결과 없음: {address}")
            return None, None

    except Exception as e:
        print(f"   [API] 호출 오류 ({address}): {e}")
        return None, None


# ---------------------------------------------------------
# 3. 좌표 관리 (캐싱 로직 포함)
# ---------------------------------------------------------
def load_coordinate_cache():
    if os.path.exists(COORD_CACHE_FILE):
        with open(COORD_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_coordinate_cache(cache):
    # data 폴더가 없으면 생성
    os.makedirs(os.path.dirname(COORD_CACHE_FILE), exist_ok=True)
    with open(COORD_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------
# 4. 메인 데이터 생성 로직
# ---------------------------------------------------------
def generate_jeonse_map_json():
    print("=" * 60)
    print("🗺️ [Map Data] 전세가율 지도 데이터 생성 시작 (with Kakao API)")
    print("=" * 60)

    # 1. 데이터 로드 (DB)
    df = load_pure_market_data()

    # 2. 지역명 매핑 (시군구 코드 -> 한글명)
    sigungu_map = get_sigungu_map_from_db()
    df['region_name'] = df['시군구'].astype(str).map(sigungu_map).fillna(df['시군구'])

    # 3. 월(Month) 컬럼 생성
    df['month'] = df['contract_date'].dt.strftime('%Y-%m')

    # 4. 그룹화 (시군구/월별 통계)
    grouped = df.groupby(['시군구', 'region_name', 'month'])['jeonse_ratio'].agg(['mean', 'count']).reset_index()

    # 5. 좌표 캐시 로드
    coord_cache = load_coordinate_cache()

    # 6. 최종 리스트 생성
    result_list = []
    unique_regions = grouped[['시군구', 'region_name']].drop_duplicates()

    print(f">> 총 {len(unique_regions)}개 지역에 대한 좌표 매핑 시작...")

    for _, row in unique_regions.iterrows():
        code = str(row['시군구'])
        name = row['region_name']

        # (1) 좌표 구하기 (캐시 확인 -> 없으면 API 호출)
        if code in coord_cache:
            lat, lng = coord_cache[code]
        else:
            # 시군구 이름으로 검색 (예: "서울 종로구")
            # 검색 정확도를 위해 '청'이나 '시청' 등을 붙일 수도 있지만, 행정구역명 자체로도 잘 나옴
            search_query = name
            lat, lng = get_coordinates_from_kakao(search_query)

            if lat and lng:
                coord_cache[code] = [lat, lng]
                time.sleep(0.1)  # API 속도 제한 방지 (0.1초 대기)
                print(f"   [API] 좌표 획득 완료: {name} -> {lat}, {lng}")
            else:
                # 좌표 못 찾으면 기본값 (서울시청 근처 등) 또는 제외
                lat, lng = 37.5665, 126.9780

                # (2) 해당 지역의 월별 데이터 필터링
        region_data = grouped[grouped['시군구'] == code].sort_values('month')

        history_list = []
        total_count = 0

        for _, h_row in region_data.iterrows():
            history_list.append({
                "month": h_row['month'],
                "ratio": round(h_row['mean'], 1),
                "count": int(h_row['count'])
            })
            total_count += int(h_row['count'])

        # (3) 요약 정보 생성
        latest = history_list[-1] if history_list else {'ratio': 0}
        latest_ratio = latest['ratio']

        risk_level = "SAFE"
        if latest_ratio >= 80:
            risk_level = "RISKY"
        elif latest_ratio >= 70:
            risk_level = "CAUTION"

        # (4) 결과 객체 조립
        result_list.append({
            "region_code": code,
            "region_name": name,
            "coordinates": {
                "lat": lat,
                "lng": lng
            },
            "summary": {
                "latest_ratio": latest_ratio,
                "risk_level": risk_level,
                "total_tx_count": total_count
            },
            "history": history_list
        })

    # 7. 캐시 및 결과 저장
    save_coordinate_cache(coord_cache)

    output_path = os.path.join(project_root, 'models', 'map_data_final.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 지도 데이터 생성 완료: {output_path}")
    print(f"   (캐시된 지역 수: {len(coord_cache)}개)")

    return result_list


if __name__ == "__main__":
    if KAKAO_API_KEY == "YOUR_KAKAO_REST_API_KEY_HERE":
        print("❌ 오류: KAKAO_API_KEY를 설정해주세요!")
    else:
        generate_jeonse_map_json()