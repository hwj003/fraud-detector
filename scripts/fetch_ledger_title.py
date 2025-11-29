import os, time, sys, re
from dotenv import load_dotenv
import requests
import json
import sqlite3
import urllib.parse
from db_manager import init_db, get_connection
from kakao_localmap_api import get_building_name_from_kakao
load_dotenv()
# 전유부 (호수별) 데이터 수집 (가격, 소유자)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'local_fraud_db.sqlite'))

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CLIENT_ID = os.getenv("CLIENT_ID_2")
CLIENT_SECRET = os.getenv("CLIENT_SECRET_2")
CODEF_USER_ID = os.getenv("CODEF_USER_ID_2")
CODEF_USER_RSA_PASSWORD = os.getenv("CODEF_USER_RSA_PASSWORD_2")

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


def collect_title_data(token, start_address):
    """
    2-Way 방식 표제부 수집 로직
    """
    # 0. 주소 변환
    building_name = get_building_name_from_kakao(start_address)
    target_address = f"{start_address} {building_name}"
    print(f"   [Request] 표제부 조회 시작: {target_address}")

    # 1. Step 1 호출 (주소 검색)
    res_step1 = fetch_step1_search(token, target_address)

    if not res_step1:
        print("   [Fail] Step 1 응답 없음")
        return

    code = res_step1['result']['code']
    data = res_step1.get('data', {})

    # ------------------------------------------------------------------
    # Case A: 바로 결과가 나온 경우 (단일 건물, 동 선택 불필요)
    # ------------------------------------------------------------------
    if code == 'CF-00000':
        print("   [Info] 단일 건물 표제부 발견 (즉시 저장)")
        parse_and_save_title(res_step1, start_address)
        return

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
            save_job_log(start_address, status="DATA_NOT_FOUND")
            return

        print(f"   [Info] {len(dong_list)}개 동 발견. 상세 수집 시작...")

        valid_dongs = []
        skip_keywords = ['상가', '근린', '경비실', '주차장', '기계실', '관리동', '노인정', '유치원', '커뮤니티']

        # 1. 필터링 (비주거용 제외)
        for d in dong_list:
            d_name = urllib.parse.unquote_plus(d.get('reqDong', '')).strip()
            if any(k in d_name for k in skip_keywords):
                continue
            valid_dongs.append(d)

        if not valid_dongs:
            print("   [Skip] 수집할 주거용 동이 없습니다. (상가단지 등)")
            save_job_log(start_address, status="NO_RESIDENTIAL_DONG")
            return

        # 2. 정렬 및 중간값 선택
        # 동 이름 기준으로 정렬 (101동, 102동...)
        valid_dongs.sort(key=lambda x: x.get('reqDong'))

        mid_idx = len(valid_dongs) // 2
        target_dong = valid_dongs[mid_idx]  # 표본 동 선택!

        target_dong_name = urllib.parse.unquote_plus(target_dong.get('reqDong', '')).strip()
        target_dong_num = target_dong.get('commDongNum')

        print(f"   [Selected] 총 {len(valid_dongs)}개 동 중 표본 수집: '{target_dong_name}'")

        # 3. 선택된 동만 상세 조회 (Step 2 호출)
        res_step2 = fetch_step2_detail(
            token, jti, job_index, thread_index, two_way_timestamp, target_dong_num, target_address
        )

        if res_step2 and res_step2['result']['code'] == 'CF-00000':
            # 저장 함수 호출
            parse_and_save_title(res_step2, start_address)

            print(f"   [Done] '{target_dong_name}' 표제부 수집 완료.")
            return
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
    else:
        msg = res_step1['result']['message']
        print(f"   [Error] API 오류 ({code}): {msg}")
        save_job_log(start_address, status=code)

# ==========================================
# 5. 실행부 (테스트용)
# ==========================================
if __name__ == "__main__":
    token = get_access_token()

    if token:
        target_list = get_targets_from_exclusive_db(limit=100)

        if not target_list:
            print("모든 데이터가 최신이거나, 수집할 대상이 없습니다.")

        for idx, target_addr in enumerate(target_list):
            print(f"\n===============================================================")
            print(f"[진행률 {idx + 1}/{len(target_list)}] Target: {target_addr}")
            print(f"===============================================================")

            # 표제부 데이터 수집
            collect_title_data(token, target_addr)

            # 건물이 바뀔 때마다 잠시 휴식 (API 보호)
            time.sleep(1)

    else:
        print("토큰 발급 실패. 설정을 확인하세요.")