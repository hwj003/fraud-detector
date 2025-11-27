import requests
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

def get_road_address_from_kakao(jibun_address):
    """
    지번 주소 -> 도로명 주소 변환 함수
    """
    api_key = os.getenv("KAKAO_API_KEY")
    if not api_key:
        print("KAKAO_API_KEY가 설정되지 않았습니다.")
        return None

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": jibun_address}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()

        result = response.json()
        documents = result.get('documents', [])

        if documents:
            # road_address가 있으면 반환, 없으면(지번만 있는 땅) None
            road_addr_obj = documents[0].get('road_address')
            if road_addr_obj:
                full_addr = road_addr_obj.get('address_name', '')
                bldg_name = road_addr_obj.get('building_name', '').strip()

                if bldg_name:
                    return f"{full_addr} {bldg_name}"
                else:
                    return full_addr

        return None

    except Exception as e:
        print(f"[Kakao] API 오류: {e}")
        return None

def get_building_name_from_kakao(jibun_address):
    """
    해당 지번의 건물명을 가져옵니다.
    """
    api_key = os.getenv("KAKAO_API_KEY")
    if not api_key:
        print("KAKAO_API_KEY가 설정되지 않았습니다.")
        return None

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": jibun_address}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()

        result = response.json()
        documents = result.get('documents', [])

        if documents:
            # road_address가 있으면 반환, 없으면(지번만 있는 땅) None
            road_addr_obj = documents[0].get('road_address')
            if road_addr_obj:
                bldg_name = road_addr_obj.get('building_name', '').strip()

                if bldg_name:
                    return bldg_name
                else:
                    return ""

        return ""

    except Exception as e:
        print(f"[Kakao] API 오류: {e}")
        return None

if __name__ == "__main__":
    input_address = "인천광역시 부평구 부개동 402"
    target_address = get_building_name_from_kakao(input_address)
    # 광일아파트
    print(target_address)