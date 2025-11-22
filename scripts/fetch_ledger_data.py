import pandas as pd
import os
from dotenv import load_dotenv
import requests
import json
import sqlite3
import urllib.parse
import base64

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 목표: scripts 폴더의 '상위 폴더'에 있는 local_fraud_db.sqlite
# 만약 같은 폴더에 있다면 '..'을 빼고 파일명만 적으세요.
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'local_fraud_db.sqlite'))

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# API 엔드포인트
TOKEN_URL = "https://oauth.codef.io/oauth/token"
API_URL = "https://development.codef.io/v1/kr/public/lt/eais/general-buildings"

# ==========================================
# 2. 헬퍼 함수: 데이터 디코딩 및 유틸
# ==========================================
def decode_value(value):
    """
    URL 인코딩된 문자열을 디코딩합니다.
    예: '서울특별시+강남구' -> '서울특별시 강남구'
    """
    if isinstance(value, str):
        # unquote_plus는 %XX 디코딩과 '+'를 공백으로 변환을 동시에 수행
        return urllib.parse.unquote_plus(value)
    return value

def recursive_decode(data):
    """
    딕셔너리나 리스트를 순회하며 모든 문자열 값을 디코딩합니다.
    """
    if isinstance(data, dict):
        return {k: recursive_decode(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_decode(i) for i in data]
    else:
        return decode_value(data)

# ==========================================
# 3. Codef API 로직
# ==========================================
def get_access_token(client_id, client_secret):
    """
    OAuth 2.0 토큰 발급
    """

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {"grant_type": "client_credentials", "scope": "read"}  # Scope는 문서 참조 필요

    try:
        response = requests.post(
            TOKEN_URL,
            headers=headers,
            data=data,
            auth=(client_id, client_secret)
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"토큰 발급 실패: {e}")
        return None

def fetch_building_ledger(access_token, address):
    """
    건축물대장 API 호출
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # 요청 바디
    payload = {
        "organization": "0008",
        "loginType": "1",
        "userId": os.getenv("CODEF_USER_ID"),
        "userPassword": os.getenv("CODEF_USER_RSA_PASSWORD"),
        "address": address,
        "dong": ""
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        try:
            return response.json()
        except json.JSONDecodeError:
            # 2. 실패하면(지금 같은 상황), URL 디코딩 후 다시 JSON 변환 시도
            print("응답이 URL 인코딩되어 있습니다. 디코딩을 시도합니다.")
            # %7B -> { 로 변환
            decoded_str = urllib.parse.unquote_plus(response.text)
            # 문자열을 JSON 객체(Dictionary)로 변환
            return json.loads(decoded_str)

    except Exception as e:
        print(f"API 호출 실패: {e}")
        return None


# ==========================================
# 2. 전세사기 핵심 데이터 파싱 함수 (문제 3 해결)
# ==========================================
def process_and_save(api_json):
    """
    건축물대장 JSON에서 전세사기 예측에 필요한 핵심 피처만 추출
    """
    if api_json.get('result', {}).get('code') != 'CF-00000':
        return None, None, None

    data = api_json['data']

    # -----------------------------------------------
    # A. 건물 기본 정보 & 위반 여부 (Risk Features)
    # -----------------------------------------------
    # resDetailList를 딕셔너리로 변환 (검색 용이성)
    details_list = data.get('resDetailList', [])
    details_dict = {}
    for item in details_list:
        key = item['resType'].replace('※', '').replace(' ', '')
        # URL 디코딩 및 특수문자 제거
        val = urllib.parse.unquote_plus(item['resContents'])
        val = val.replace(',', '').replace('+', '').replace('㎡', '').replace('%', '')
        details_dict[key] = val

    # 중요: 위반 건축물 이력 파싱 (resChangeList 분석)
    # "위반"이라는 키워드가 포함된 변경 이력을 모두 텍스트로 수집
    change_list = data.get('resChangeList', [])
    violation_history = []
    for change in change_list:
        reason = urllib.parse.unquote_plus(change.get('resChangeReason', ''))
        if '위반' in reason:
            violation_history.append(f"[{change['resChangeDate']}] {reason}")

    violation_text = " | ".join(violation_history) if violation_history else "없음"

    # 현재 위반 상태
    current_violation = data.get('resViolationStatus', '')
    is_violated = 'Y' if current_violation or violation_history else 'N'

    # 메인 데이터프레임 생성
    main_info = {
        'unique_no': data.get('commUniqeNo'),  # 고유번호 (Join Key)
        'road_addr': urllib.parse.unquote_plus(data.get('commAddrRoadName', '')),
        'lot_addr': urllib.parse.unquote_plus(data.get('resUserAddr', '')),
        'main_usage': details_dict.get('주용도', ''),  # 근생 빌라 판별용
        'total_area': details_dict.get('연면적', 0),  # 시세 추정용
        'approval_date': details_dict.get('사용승인일', ''),  # 노후도(깡통전세)
        'is_violation': is_violated,  # [핵심] 위반건축물 여부 (과거포함)
        'violation_details': violation_text,  # [핵심] 위반 상세 내용
        'owner_name': '',  # 아래에서 채움 (신탁 여부 확인용)
        'ownership_stake': ''  # 아래에서 채움 (지분 쪼개기 확인용)
    }

    # -----------------------------------------------
    # B. 소유자 정보 (권리 관계 분석)
    # -----------------------------------------------
    # 소유자 리스트에서 대표 소유자 및 지분 정보 추출
    owner_list = data.get('resOwnerList', [])
    if owner_list:
        # 첫 번째 소유자 정보만 예시로 가져옴 (필요시 별도 테이블 분리)
        # 신탁 회사가 껴있는지 확인하기 위해 이름 중요
        owner_info = owner_list[0]
        main_info['owner_name'] = urllib.parse.unquote_plus(owner_info.get('resOwner', ''))
        main_info['ownership_stake'] = urllib.parse.unquote_plus(owner_info.get('resOwnershipStake', ''))

    df_main = pd.DataFrame([main_info])

    # -----------------------------------------------
    # C. 층별 용도 정보 (근생/주거 불법 개조 확인)
    # -----------------------------------------------
    status_list = data.get('resBuildingStatusList', [])
    if status_list:
        df_floors = pd.DataFrame(status_list)
        df_floors['commUniqeNo'] = data.get('commUniqeNo')  # FK

        # 데이터 정제
        df_floors = df_floors.map(lambda x: urllib.parse.unquote_plus(x) if isinstance(x, str) else x)

        # 컬럼명 통일 및 필요한 컬럼만 선택
        df_floors = df_floors.rename(columns={
            'resFloor': 'floor',
            'resUseType': 'usage',  # 이 컬럼이 '사무소'인데 실제 집으로 쓰면 불법
            'resArea': 'area'
        })[['commUniqeNo', 'floor', 'usage', 'area']]
    else:
        df_floors = pd.DataFrame()

    return df_main, df_floors


# ==========================================
# 3. DB 저장 로직 (기존 DB에 Append)
# ==========================================
def save_fraud_data(df_main, df_floors):
    conn = sqlite3.connect(DB_PATH)
    try:
        df_main.to_sql('building_risk_factors', conn, if_exists='append', index=False)

        if not df_floors.empty:
            df_floors.to_sql('building_floor_details', conn, if_exists='append', index=False)

        print(f"[DB 저장 완료] {df_main['road_addr'][0]}")

    except Exception as e:
        print(f"DB 저장 실패: {e}")
    finally:
        conn.close()


# ==========================================
# 4. 메인 실행
# ==========================================
if __name__ == "__main__":
    # 1. 검색할 주소 설정 (카카오 대신 직접 입력)
    TARGET_ADDRESS = "인천광역시 부평구 장제로 94"

    # 2. 토큰 획득
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)

    if token:
        # 3. 실제 API 호출
        api_result = fetch_building_ledger(token, TARGET_ADDRESS)

        if api_result:
            # 4. 데이터 정제 (전세사기 리스크 요인 추출)
            df_main, df_floors = process_and_save(api_result)

            # 5. DB 저장
            if df_main is not None:
                save_fraud_data(df_main, df_floors)
            else:
                print("저장할 유효한 데이터가 없습니다.")
    else:
        print("토큰이 없어 프로세스를 중단합니다.")