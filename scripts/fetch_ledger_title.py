import os, time, sys, re
from dotenv import load_dotenv
import requests
import json
import sqlite3
import urllib.parse
# [수정] 경로 문제 해결을 위한 조건부 임포트
try:
    # 1. 외부(predict.py 등)에서 패키지로 불러올 때 (프로젝트 루트 기준)
    from scripts.db_manager import init_db, get_connection
    from scripts.kakao_localmap_api import get_building_name_from_kakao, get_road_address_from_kakao
except ModuleNotFoundError:
    # 2. 이 파일을 직접 실행할 때 (현재 폴더 기준)
    from db_manager import init_db, get_connection
    from kakao_localmap_api import get_building_name_from_kakao, get_road_address_from_kakao
load_dotenv()
# 전유부 (호수별) 데이터 수집 (가격, 소유자)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'local_fraud_db.sqlite'))

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CLIENT_ID = os.getenv("CLIENT_ID_1")
CLIENT_SECRET = os.getenv("CLIENT_SECRET_1")
CODEF_USER_ID = os.getenv("CODEF_USER_ID_1")
CODEF_USER_RSA_PASSWORD = os.getenv("CODEF_USER_RSA_PASSWORD_1")

# API 엔드포인트
TOKEN_URL = "https://oauth.codef.io/oauth/token"
API_URL = "https://development.codef.io/v1/kr/public/lt/eais/building-ledger-heading"

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def save_job_log(address, status="TITLE_SAVED"):
    """
    작업 로그 저장 (job_type='TITLE'로 구분)
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        # job_type을 'TITLE'로 지정하여 전유부 수집과 구분
        cur.execute("""
            INSERT INTO api_job_log (search_address, job_type, status, created_at, updated_at) 
            VALUES (?, 'TITLE', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(search_address, job_type) DO UPDATE SET
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
        """, (address, status))
        conn.commit()
        print(f"      [Log Saved] '{address}' 표제부 수집 완료 ({status})")
    except Exception as e:
        print(f"      [Log Error] 로그 저장 실패: {e}")
    finally:
        conn.close()

# ==========================================
# 3. API 호출 함수
# ==========================================
def get_access_token():
    # (기존 코드와 동일)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "scope": "read"}
    try:
        response = requests.post(TOKEN_URL, headers=headers, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print(f"토큰 발급 실패: {e}")
        return None


def fetch_step1_search(token, address):
    """
    [Step 1] 주소 검색 (세션 시작)
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "organization": "0008",
        "loginType": "1",
        "userId": CODEF_USER_ID,
        "userPassword": CODEF_USER_RSA_PASSWORD,
        "address": address
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        try:
            return resp.json()
        except:
            return json.loads(urllib.parse.unquote_plus(resp.text))
    except Exception as e:
        print(f"   [API Error] Step 1 실패: {e}")
        return None


def fetch_step2_detail(token, jti, job_index, thread_index, two_way_timestamp, dong_num, address):
    """
    [Step 2] 동 코드(dongNum)를 이용한 상세 표제부 조회
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "organization": "0008",
        "loginType": "1",
        "userId": CODEF_USER_ID,
        "userPassword": CODEF_USER_RSA_PASSWORD,
        "address": address,
        "is2Way": True,
        "twoWayInfo": {
            "jobIndex": job_index,
            "threadIndex": thread_index,
            "jti": jti,
            "twoWayTimestamp": two_way_timestamp
        },
        "dongNum": dong_num  # [핵심] 여기에 동 코드를 넣어야 함
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        try:
            return resp.json()
        except:
            return json.loads(urllib.parse.unquote_plus(resp.text))
    except Exception as e:
        print(f"   [API Error] Step 2 실패 (dongNum={dong_num}): {e}")
        return None


# ==========================================
# 4. 데이터 파싱 및 DB 저장 (핵심)
# ==========================================
def parse_and_save_title(api_json, input_address):
    data = api_json.get('data', {})
    if not data:
        print(f"      [Skip] 데이터 없음: {input_address}")
        save_job_log(input_address, status="DATA_NOT_FOUND")
        return

    # 1. 기본 식별 정보 파싱
    unique_no = data.get('commUniqeNo')  # 예: 2823710700-3-04020000

    # 고유번호 파싱하여 시군구/법정동/번지 채우기
    sigungu_code = ""
    bjdong_code = ""
    bunji = ""

    if unique_no and '-' in unique_no:
        parts = unique_no.split('-')
        if len(parts) >= 3:
            code_part = parts[0]  # 2823710700
            bunji_part = parts[2]  # 04020000

            if len(code_part) >= 10:
                sigungu_code = code_part[:5]
                bjdong_code = code_part[5:10]

            if len(bunji_part) >= 8:
                bon = bunji_part[:4].lstrip('0') or '0'  # 앞의 0 제거
                bu = bunji_part[4:].lstrip('0') or '0'
                bunji = f"{bon}-{bu}" if bu != '0' else bon

    road_addr = urllib.parse.unquote_plus(data.get('commAddrRoadName', '') or '')
    detail_addr = urllib.parse.unquote_plus(data.get('reqDong', '') or '')  # 아파트명

    # 동 명칭 추출 (입력 주소에서 추출하거나, reqDong 사용)
    # reqDong이 "광일아파트"처럼 아파트명인 경우도 있고 "101동"인 경우도 있음.
    # 여기서는 일단 detail_address와 동일하게 저장하거나 별도 로직 필요
    dong_name = detail_addr

    # 2. resDetailList 파싱
    details = {}
    for item in data.get('resDetailList', []):
        key = item.get('resType', '').replace('※', '').replace(' ', '')
        val = urllib.parse.unquote_plus(item.get('resContents', '') or '')
        details[key] = val

    main_use = details.get('주용도', '알수없음')
    structure_type = details.get('주구조', '알수없음')

    # 연면적
    total_area_str = details.get('연면적', '0').replace('열', '').replace('㎡', '').replace(',', '').strip()
    try:
        total_floor_area = float(total_area_str)
    except:
        total_floor_area = 0.0

    # 세대수 파싱
    req_ho_str = urllib.parse.unquote_plus(data.get('reqHo', '') or '')
    household_cnt = 0
    match = re.search(r'(\d+)세대', req_ho_str)
    if match:
        household_cnt = int(match.group(1))
    else:
        match = re.search(r'(\d+)가구', req_ho_str)
        if match: household_cnt = int(match.group(1))

    # 층수 파싱
    floor_str = details.get('층수', '')
    grnd_flr_cnt = 0
    und_flr_cnt = 0
    match_grnd = re.search(r'지상[:\s]*(\d+)층', floor_str)
    if match_grnd: grnd_flr_cnt = int(match_grnd.group(1))
    match_und = re.search(r'지하[:\s]*(\d+)층', floor_str)
    if match_und: und_flr_cnt = int(match_und.group(1))

    # 사용승인일 파싱 강화 (1985.1.15. -> 1985-01-15)
    use_apr_day_raw = details.get('사용승인일', '').strip()
    formatted_date = None
    if use_apr_day_raw:
        # 숫자만 추출 (1985, 1, 15)
        dates = re.findall(r'\d+', use_apr_day_raw)
        if len(dates) >= 3:
            year, month, day = dates[0], dates[1], dates[2]
            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"  # 0 채우기

    is_violating = 'Y' if data.get('resViolationStatus') else 'N'

    # 주차장 대수 합산
    parking_cnt = 0
    for p_item in data.get('resParkingLotStatusList', []):
        try:
            cnt = int(p_item.get('resNumber', '0') or '0')
            parking_cnt += cnt
        except:
            pass

    # 승강기 대수 합산 (resDetailList 내부 '승강기|...' 항목 찾기)
    elevator_cnt = 0
    for key, val in details.items():
        if '승강기' in key:
            # 값에서 숫자 추출 (예: "2대" -> 2)
            nums = re.findall(r'\d+', val)
            if nums:
                elevator_cnt += int(nums[0])

    # 5. DB 저장
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO building_title_info (
                unique_number, sigungu_code, bjdong_code, bunji,
                road_address, detail_address, dong_name,
                main_use, structure_type, total_floor_area, 
                household_cnt, grnd_flr_cnt, und_flr_cnt, 
                parking_cnt, elevator_cnt, use_apr_day, is_violating
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(unique_number) DO UPDATE SET
                sigungu_code=excluded.sigungu_code,
                bjdong_code=excluded.bjdong_code,
                bunji=excluded.bunji,
                main_use=excluded.main_use,
                structure_type=excluded.structure_type,
                total_floor_area=excluded.total_floor_area,
                household_cnt=excluded.household_cnt,
                parking_cnt=excluded.parking_cnt,
                elevator_cnt=excluded.elevator_cnt,
                use_apr_day=excluded.use_apr_day,
                updated_at=CURRENT_TIMESTAMP
        """, (
            unique_no, sigungu_code, bjdong_code, bunji,
            road_addr, detail_addr, dong_name,
            main_use, structure_type, total_floor_area,
            household_cnt, grnd_flr_cnt, und_flr_cnt,
            parking_cnt, elevator_cnt, formatted_date, is_violating
        ))
        conn.commit()
        print(f"      [Saved] {detail_addr} 표제부 저장 완료 (세대수: {household_cnt}, 승강기: {elevator_cnt})")

        save_job_log(input_address, status="TITLE_SAVED")

    except Exception as e:
        conn.rollback()
        print(f"      [DB Error] {e}")
        save_job_log(input_address, status="DB_ERROR")
    finally:
        conn.close()

def get_targets_from_exclusive_db(limit=100):
    conn = get_connection()
    cur=conn.cursor()

    print("DB에서 실거래가 기반 수집 대상 추출 중...")

    # ---------------------------------------------------------
    # 쿼리 설명:
    # 수집된 건축물대장 전유부 데이터를 기반으로 주소 선택
    # ---------------------------------------------------------
    query = """
        SELECT DISTINCT 
            m.bjdong_name, 
            r.본번, 
            r.부번
        FROM raw_rent r
        JOIN meta_bjdong_codes m 
          ON r.시군구 = m.sgg_code AND r.법정동 = m.bjdong_code
        -- [조건 1] 이미 '전유부(EXCLUSIVE)' 수집은 완료된 애들만 골라라 (교집합)
        JOIN api_job_log exclusive_log 
          ON (
              m.bjdong_name || ' ' || CAST(r.본번 AS INTEGER) || 
              CASE WHEN CAST(r.부번 AS INTEGER) = 0 THEN '' ELSE '-' || CAST(r.부번 AS INTEGER) END
          ) = exclusive_log.search_address
          AND exclusive_log.job_type = 'EXCLUSIVE'
          
        -- [조건 2] 하지만 '표제부(TITLE)' 수집은 아직 안 한 애들 (차집합)
        LEFT JOIN api_job_log title_log 
          ON (
              m.bjdong_name || ' ' || CAST(r.본번 AS INTEGER) || 
              CASE WHEN CAST(r.부번 AS INTEGER) = 0 THEN '' ELSE '-' || CAST(r.부번 AS INTEGER) END
          ) = title_log.search_address
          AND title_log.job_type = 'TITLE'
          
        WHERE title_log.search_address IS NULL
        LIMIT ?
        """

    try:
        cur.execute(query, (limit,))
        rows = cur.fetchall()

        address_list = []
        for row in rows:
            bjdong_name = row[0]  # 예: 인천광역시 부평구 산곡동
            bonbeon = int(row[1])  # 0142 -> 142 (0제거)
            bubeon = int(row[2])  # 0003 -> 3

            # 주소 조립: "인천광역시 부평구 산곡동 142-3"
            if bubeon == 0:
                addr = f"{bjdong_name} {bonbeon}"
            else:
                addr = f"{bjdong_name} {bonbeon}-{bubeon}"

            address_list.append(addr)

        print(f"수집 대상 {len(address_list)}건 확보")
        return address_list
    except Exception as e:
        print(f"DB 조회 실패: {e}")
        # 테이블이 없을 경우를 대비한 안내
        print("   (참고: raw_rent 또는 meta_bjdong_codes 테이블이 존재하는지 확인하세요)")
        return []
    finally:
        conn.close()


def collect_title_data(token, start_address, base_addr):
    """
    2-Way 방식 표제부 수집 로직
    """
    print(f"   [Request] 표제부 조회 시작: {start_address}")

    # 1. Step 1 호출 (주소 검색)
    res_step1 = fetch_step1_search(token, start_address)

    if not res_step1:
        print("   [Fail] Step 1 응답 없음")
        return True

    code = res_step1['result']['code']
    data = res_step1.get('data', {})

    # ------------------------------------------------------------------
    # Case A: 바로 결과가 나온 경우 (단일 건물, 동 선택 불필요)
    # ------------------------------------------------------------------
    if code == 'CF-00000':
        print("   [Info] 단일 건물 표제부 발견 (즉시 저장)")
        parse_and_save_title(res_step1, start_address)
        return True

    # ------------------------------------------------------------------
    # Case B: 추가 입력 필요 (CF-03002) -> 동 목록이 온 경우
    # ------------------------------------------------------------------
    elif code == 'CF-03002':
        # 세션 정보 추출
        jti = data.get('jti')
        job_index = data.get('jobIndex')
        thread_index = data.get('threadIndex')
        two_way_timestamp = data.get('twoWayTimestamp')

        # 동 목록 추출 (extraInfo 내부에 있음)
        extra_info = data.get('extraInfo', {})
        dong_list = extra_info.get('reqDongNumList', [])

        if not dong_list:
            print("   [Skip] 동 목록이 비어 있습니다.")
            save_job_log(base_addr, status="DATA_NOT_FOUND")
            return False

        print(f"   [Info] {len(dong_list)}개 동 발견. 상세 수집 시작...")

        valid_dongs = []
        skip_keywords = ['상가', '근린', '경비실', '주차장', '기계실', '관리동', '노인정', '유치원', '커뮤니티']

        # 1. 필터링 (비주거용 제외)
        for d in dong_list:
            # reqDong이 공란일 경우 빈값으로 처리
            d_val = d.get('reqDong') or ''
            d_name = urllib.parse.unquote_plus(d_val).strip()
            if any(k in d_name for k in skip_keywords):
                continue
            valid_dongs.append(d)

        if not valid_dongs:
            print("   [Skip] 수집할 주거용 동이 없습니다. (상가단지 등)")
            save_job_log(base_addr, status="NO_RESIDENTIAL_DONG")
            return False

        # 2. 정렬 및 중간값 선택
        # 동 이름 기준으로 정렬 (101동, 102동...)
        valid_dongs.sort(key=lambda x: (x.get('reqDong') or ""))

        mid_idx = len(valid_dongs) // 2
        target_dong = valid_dongs[mid_idx]  # 표본 동 선택!

        target_dong_name = urllib.parse.unquote_plus(target_dong.get('reqDong', '')).strip()
        target_dong_num = target_dong.get('commDongNum')

        print(f"   [Selected] 총 {len(valid_dongs)}개 동 중 표본 수집: '{target_dong_name}'")

        # 3. 선택된 동만 상세 조회 (Step 2 호출)
        res_step2 = fetch_step2_detail(
            token, jti, job_index, thread_index, two_way_timestamp, target_dong_num, start_address
        )

        if res_step2 and res_step2['result']['code'] == 'CF-00000':
            # 저장 함수 호출
            parse_and_save_title(res_step2, start_address)

            print(f"   [Done] '{target_dong_name}' 표제부 수집 완료.")
            return True
        else:
            err_msg = res_step2['result']['message'] if res_step2 else 'Error'
            print(f"   [Fail] 수집 실패: {err_msg}")
            # 실패해도 일단 로그는 남기거나, 재시도를 위해 안 남길 수도 있음. 여기선 재시도 위해 로그 안 남김.

    # ------------------------------------------------------------------
    # Case C: 에러
    # ------------------------------------------------------------------
    elif code == 'CF-00012':
        print("100회 제한 초과 (CF-00012)")
        sys.exit(0)
    elif code == 'CF-13006':
        msg = res_step1['result']['message']
        print(f"   [Error] API 오류 ({code}): {msg}")
        return False
    else:
        msg = res_step1['result']['message']
        print(f"   [Error] API 오류 ({code}): {msg}")
        return False

def _collect_title_with_retry(token, address):
    """
    [Internal] 표제부 수집 실행 (지번 시도 -> 실패시 도로명 재시도)
    성공 시 True, 실패 시 False 반환
    """
    print(f"      [Work] 표제부(Title) 수집 시작...")

    # 1차 시도: 입력받은 지번 주소로 시도
    if collect_title_data(token, address, address):
        return True

    # 2차 시도: 도로명 주소 + 건물명 조합으로 재시도
    try:
        road_part = get_road_address_from_kakao(address)
        build_part = get_building_name_from_kakao(address)
        retry_address = f"{road_part} {build_part}".strip()

        print(f"      [Retry] 표제부: 번지 실패 -> 도로명 재시도: {retry_address}")
        if collect_title_data(token, retry_address, address):
            return True
    except Exception as e:
        print(f"      [Error] 표제부 재시도 주소 생성 실패: {e}")

    return False


import pandas as pd
import os
import sys
import time
from sqlalchemy import text
from tqdm import tqdm  # 진행률 표시 라이브러리 (없으면 pip install tqdm)

# --- 프로젝트 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from app.core.config import engine
from scripts.fetch_ledger_title import collect_title_data  # 표제부 수집 함수 임포트
from scripts.fetch_ledger_exclusive import get_access_token  # 토큰 발급 함수


def fetch_missing_titles():
    print("--- [Start] 표제부(Title) 누락 데이터 수집 시작 ---")

    # 1. 누락된 PNU 조회
    print(">> 1. 누락 데이터 조회 중...")

    query = """
        SELECT DISTINCT 
            SUBSTR(b.unique_number, 1, 10) as bjd, 
            SUBSTR(b.unique_number, 14, 8) as bunji, 
            MAX(b.lot_address) as address,                -- API 호출용 지번 주소
            MAX(b.road_address) as road_address           -- API 호출용 도로명 주소
        FROM building_info b
        LEFT JOIN building_title_info t 
            ON SUBSTR(b.unique_number, 1, 21) = t.unique_number
        WHERE t.unique_number IS NULL       -- 표제부에 없는 경우
          AND b.unique_number IS NOT NULL   
          AND LENGTH(b.unique_number) >= 19 -- 유효한 PNU 길이 확인
          AND SUBSTR(b.unique_number, 1, 5) = '28237'
        GROUP BY SUBSTR(b.unique_number, 1, 19)
    """

    try:
        df_missing = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ DB 조회 실패: {e}")
        return

    total_cnt = len(df_missing)
    if total_cnt == 0:
        print("✅ 모든 데이터가 표제부를 가지고 있습니다. (누락 없음)")
        return

    print(f"-> 총 {total_cnt}건의 건물 표제부가 누락되었습니다.")
    print(">> 2. API 수집 시작...")

    # 2. 토큰 발급
    token = get_access_token()
    if not token:
        print("❌ API 토큰 발급 실패. 종료합니다.")
        return

    success_cnt = 0
    fail_cnt = 0

    # 3. 순회하며 수집
    # tqdm을 사용하여 진행바 표시
    for idx, row in tqdm(df_missing.iterrows(), total=total_cnt, desc="Collecting"):
        bjd = row['bjd']
        bunji = row['bunji']

        target_addr=convert_code_to_address(bjd, bunji)
        try:
            # 표제부 수집 함수 호출 (기존 모듈 활용)
            _collect_title_with_retry(token, target_addr)

            # API 부하 방지를 위한 미세 딜레이 (필요 시 조절)
            time.sleep(0.1)

        except Exception as e:
            print(f"\n[Error]({target_addr}) 처리 중 오류: {e}")
            fail_cnt += 1

    print("\n" + "=" * 50)
    print(f"🏁 수집 완료")
    print(f"   - 대상: {total_cnt}건")
    print(f"   - 성공: {success_cnt}건")
    print(f"   - 실패: {fail_cnt}건")
    print("=" * 50)


def convert_code_to_address(bjd, bunji):
    """
    입력: "2823710100 00100272" (법정동코드10자리 + 지번8자리)
    출력: "인천광역시 부평구 부평동 10-272"
    동작: meta_bjdong_codes 테이블을 조회하여 주소명을 완성함
    """
    try:

        # 2. 시군구/법정동 코드 분리
        sgg_code = bjd[0:5]  # '2823710100'
        bjdong_code= bjd[5:10]
        # 3. DB 조회 (meta_bjdong_codes 테이블)
        # 컬럼명이 sgg_name, bjdong_name 이라고 가정합니다.
        # 실제 테이블의 컬럼명에 맞춰 수정해주세요 (예: 법정동명 등)
        query = text("""
            SELECT bjdong_name
            FROM meta_bjdong_codes
            WHERE sgg_code = :sgg_code 
              AND bjdong_code = :bjdong_code
            LIMIT 1
        """)

        with engine.connect() as conn:
            result = conn.execute(query, {"sgg_code": sgg_code, "bjdong_code": bjdong_code}).fetchone()

        if not result:
            return f"주소 정보 없음 (Code: {sgg_code})"

        # 4. 주소 문자열 조합 (None 방지 처리)
        region_name = result.bjdong_name.strip()  # 공백 제거

        # 5. 본번/부번 파싱 (00100272 -> 10-272)
        if len(bunji) == 8:
            bon = int(bunji[:4])
            bu = int(bunji[4:])

            if bu > 0:
                jibun = f"{bon}-{bu}"
            else:
                jibun = f"{bon}"
        else:
            jibun = bunji

        # 6. 최종 반환
        return f"{region_name} {jibun}"

    except Exception as e:
        return f"변환 중 오류 발생: {e}"

if __name__ == "__main__":
    fetch_missing_titles()