import pandas as pd
import os, time, sys, re
from dotenv import load_dotenv
import requests
import json
import sqlite3
import urllib.parse
import base64
from db_manager import init_db, get_connection
from kakao_localmap_api import get_building_name_from_kakao
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'local_fraud_db.sqlite'))

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CLIENT_ID = os.getenv("CLIENT_ID_1")
CLIENT_SECRET = os.getenv("CLIENT_SECRET_1")

# API 엔드포인트
TOKEN_URL = "https://oauth.codef.io/oauth/token"
API_URL = "https://development.codef.io/v1/kr/public/lt/eais/aggregate-buildings"


def extract_floor_from_ho_name(ho_name):
    """
    호 명칭에서 층수를 추출하는 헬퍼 함수
    예: '1503호' -> 15, '101호' -> 1
    """
    numbers = re.findall(r'\d+', ho_name)
    if not numbers:
        return -1

    val = int(numbers[0])

    # 3자리 이하 (101~909) -> 100으로 나눈 몫
    # 4자리 이상 (1001~3505) -> 100으로 나눈 몫
    floor = val // 100

    # 지하(B), 지층, 또는 계산된 층수가 0인 경우 제외
    if 'B' in ho_name or '지' in ho_name or floor == 0:
        return -1

    return floor

def fetch_two_way_data(token, jti, job_index, thread_index, dong_num=None, ho_num=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "organization": "0008",
        "loginType": "1",
        "is2Way": True,
        "twoWayInfo": {
            "jobIndex": job_index,
            "threadIndex": thread_index,
            "jti": jti,
            "twoWayTimestamp": int(time.time() * 1000)
        }
    }

    # 동 선택 단계
    if dong_num and not ho_num:
        payload["dongNum"] = dong_num

    # 호 선택 단계
    if dong_num and ho_num:
        payload["dongNum"] = dong_num
        payload["hoNum"] = ho_num

    try:
        # 2-Way 요청은 원래 URL로 다시 POST를 날립니다.
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"2-Way 요청 실패: {e}")
        return None

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
def get_access_token():
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
            auth=(CLIENT_ID, CLIENT_SECRET)
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"토큰 발급 실패: {e}")
        return None


def send_api_request(token, payload):
    """API 요청 전송 및 에러 핸들링 공통 함수"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()

        # URL Decoding 처리 (Codef 응답이 인코딩된 경우)
        try:
            return resp.json()
        except json.JSONDecodeError:
            decoded = urllib.parse.unquote_plus(resp.text)
            return json.loads(decoded)

    except Exception as e:
        print(f"API 요청 실패: {e}")
        return None


def fetch_initial_search(token, address):
    """[Step 1] 최초 주소 검색 요청"""

    payload = {
        "organization": "0008",
        "loginType": "1",
        "id": os.getenv("CODEF_USER_ID_1"),
        "password": os.getenv("CODEF_USER_RSA_PASSWORD_1"),
        "address": address,
        "dong": ""
    }
    return send_api_request(token, payload)


def fetch_next_step(token, jti, jobIndex, twoWayTimestamp, threadIndex, dong_num=None, ho_num=None):
    """[Step 2 & 3] 2-Way 추가 인증 요청 (동 선택 / 호 선택)"""
    payload = {
        "organization": "0008",
        "loginType": "1",
        "id": os.getenv("CODEF_USER_ID_1"),
        "password": os.getenv("CODEF_USER_RSA_PASSWORD_1"),
        "is2Way": True,
        "twoWayInfo": {
            "jobIndex": jobIndex,
            "threadIndex": threadIndex,
            "jti": jti,
            "twoWayTimestamp": twoWayTimestamp
        }
    }
    # 파라미터 동적 추가
    if dong_num: payload["dongNum"] = dong_num
    if ho_num: payload["hoNum"] = ho_num

    return send_api_request(token, payload)


# ==========================================
# 4. 데이터 파싱 및 DB 저장
# ==========================================
def parse_and_save(api_json, dong_name, ho_name):
    """
    1. 전유부(resType='0') 데이터로 건물 정보 파싱
    2. 소유자(resOwnerList) 데이터 파싱 (이름, 변동일, 원인)
    3. DB 저장 (중복 방지 및 가격 정보 동기화)
    """
    data = api_json.get('data', {})
    if not data:
        print(f"      [Skip] 데이터 없음: {dong_name} {ho_name}")
        return

    # ---------------------------------------------------------
    # 1. 건물 기본 정보 파싱 (전유부 찾기)
    # ---------------------------------------------------------
    origin_unique_no = data.get('commUniqeNo', '')
    unique_no = f"{origin_unique_no}-{ho_name}" if origin_unique_no else None

    building_id_code = data.get('resDocNo')
    road_addr = urllib.parse.unquote_plus(data.get('commAddrRoadName', '') or '')
    lot_addr = data.get('commAddrLotNumber', '') or ''
    detail_addr = f"{dong_name} {ho_name}"
    is_violating = 'Y' if data.get('resViolationStatus') else 'N'

    # [핵심] resType이 '0'(전유부분)인 항목 찾기
    exclusive_area = 0.0
    main_use = "알수없음"
    structure_type = "알수없음"

    target_item = None
    for item in data.get('resOwnedList', []):
        if item.get('resType') == '0':
            target_item = item
            break

    if target_item:
        main_use = urllib.parse.unquote_plus(target_item.get('resUseType', '') or '')
        structure_type = urllib.parse.unquote_plus(target_item.get('resStructure', '') or '')
        exclusive_area = float(target_item.get('resArea', 0))

    # 구조 정보 누락 대비 안전장치 (공용부에서라도 구조 가져오기)
    if not structure_type or structure_type == "알수없음":
        for item in data.get('resOwnedList', []):
            temp_str = urllib.parse.unquote_plus(item.get('resStructure', '') or '')
            if temp_str:
                structure_type = temp_str
                break

    # ---------------------------------------------------------
    # 2. 소유자 정보 파싱 (질문하신 부분)
    # ---------------------------------------------------------
    owner_nm = ""
    ownership_date = None
    ownership_cause = ""

    # 소유자 리스트가 존재하면 첫 번째 사람(현 소유자) 정보를 가져옴
    if data.get('resOwnerList'):
        current_owner = data['resOwnerList'][0]

        # 1. 소유자명 디코딩
        owner_nm = urllib.parse.unquote_plus(current_owner.get('resOwner', '') or '')

        # 2. 변동일자 (YYYYMMDD -> YYYY-MM-DD 변환)
        raw_date = current_owner.get('resChangeDate', '')
        if len(raw_date) == 8:
            ownership_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"

        # 3. 변동원인 (매매, 소유권이전 등)
        ownership_cause = urllib.parse.unquote_plus(current_owner.get('resChangeReason', '') or '')

    # ---------------------------------------------------------
    # 3. DB 저장 트랜잭션
    # ---------------------------------------------------------
    conn = get_connection()
    cur = conn.cursor()

    try:
        building_id = None

        # Step 1: 건물 정보 저장 (Insert or Update)
        cur.execute("""
            SELECT id FROM building_info 
            WHERE road_address = ? AND detail_address = ?
        """, (road_addr, detail_addr))

        row = cur.fetchone()

        if row:
            # Update
            building_id = row[0]
            cur.execute("""
                UPDATE building_info 
                SET unique_number=?, building_id_code=?, exclusive_area=?, 
                    main_use=?, structure_type=?, owner_name=?, 
                    ownership_changed_date=?, ownership_cause=?, is_violating_building=?
                WHERE id = ?
            """, (
                unique_no, building_id_code, exclusive_area,
                main_use, structure_type, owner_nm,
                ownership_date, ownership_cause, is_violating,
                building_id
            ))
        else:
            # Insert
            cur.execute("""
                INSERT INTO building_info (
                    unique_number, building_id_code, road_address, lot_address, detail_address,
                    exclusive_area, main_use, structure_type,
                    owner_name, ownership_changed_date, ownership_cause, is_violating_building
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                unique_no, building_id_code, road_addr, lot_addr, detail_addr,
                exclusive_area, main_use, structure_type,
                owner_nm, ownership_date, ownership_cause, is_violating
            ))
            building_id = cur.lastrowid

        # Step 2: 가격 정보 저장
        if building_id:
            cur.execute("DELETE FROM public_price_history WHERE building_info_id = ?", (building_id,))

            price_list = data.get('resPriceList', [])
            insert_data = []

            for price_item in price_list:
                raw_price = price_item.get('resBasePrice', '0').replace(',', '')
                raw_price_date = price_item.get('resReferenceDate', '')
                fmt_price_date = f"{raw_price_date[:4]}-{raw_price_date[4:6]}-{raw_price_date[6:]}" if len(
                    raw_price_date) == 8 else raw_price_date

                insert_data.append((building_id, fmt_price_date, raw_price))

            if insert_data:
                cur.executemany("""
                    INSERT INTO public_price_history (building_info_id, base_date, price)
                    VALUES (?, ?, ?)
                """, insert_data)

            conn.commit()
            print(f"      [Saved] {dong_name} {ho_name} (ID: {building_id}, 주인: {owner_nm})")
        else:
            print(f"      [Error] ID 확보 실패")

    except sqlite3.IntegrityError as e:
        conn.rollback()
        print(f"      [DB Error] {e}")
    except Exception as e:
        conn.rollback()
        print(f"      [Error] {e}")
    finally:
        conn.close()


def get_dong_list_step(token, address):
    """
    [Step 0] 주소를 입력받아 '동(Dong)' 목록을 반환
    """
    print(f"   [Step 0] 주소 검색 및 동 목록 확보: {address}")
    res = fetch_initial_search(token, address)

    if not res: return None, None

    code = res['result']['code']
    data = res.get('data', {})
    extra = data.get('extraInfo', {})

    # Case: 동 목록 리턴
    if extra.get('reqDongNumList'):
        return extra['reqDongNumList'], data  # 동 리스트, 세션정보 반환

    # Case: 바로 호 목록 리턴 (나홀로 아파트 등)
    elif extra.get('reqHoNumList'):
        # 동 정보가 없으므로 가상의 동으로 처리하거나 로직 분기 필요
        return [{'reqDong': 'default', 'commDongNum': ''}], data

    # Case: 바로 결과 리턴 (단독주택 등)
    elif code == 'CF-00000':
        return [], data  # 즉시 저장 대상
    elif code == 'CF-00012':
        print(f"일일 100회 호출 제한 횟수를 초과했습니다. {code}")
        sys.exit(0) # 프로그램 종료
    else:
        print(f"에러 발생: 에러 코드 - {code}")
        return [], res

def get_ho_list_step(token, address, dong_code):
    """
    [Step 1] 주소+동코드를 이용하여 '호(Ho)' 목록을 반환
    (주의: 호 목록만 얻고 세션은 버립니다. 실제 데이터 수집 시 새로 뚫습니다.)
    """
    # 1. 새 세션 시작 (주소 검색)
    res0 = fetch_initial_search(token, address)
    if not res0: return None

    code = res0['result']['code']
    data0 = res0.get('data', {})

    # CF-00012: 한도 초과
    if code == 'CF-00012':
        print("[오류] 일일 100회 호출 한도 초과 CF-00012")
        sys.exit(0)

    # CF-03002: 추가 입력 필요 (정상적인 경우, jti가 있음)
    if code == 'CF-03002' and 'jti' in data0:
        pass # 정상 진행
    else:
        # 그 외의 경우 (에러이거나, 단일 건물이어서 바로 결과가 나온 경우)
        # 호 목록을 뽑을 수 없는 상태이므로 빈 리스트 반환
        msg = res0['result']['message']
        print(f"      [Warning] 호 목록 진입 불가 (Code: {code}, Msg: {msg})")
        return []

    # 2. 동 선택 요청
    res1 = fetch_next_step(
        token,
        jti=data0['jti'],
        jobIndex=data0['jobIndex'],
        threadIndex=data0['threadIndex'],
        twoWayTimestamp=data0['twoWayTimestamp'],
        dong_num=dong_code
    )

    if not res1: return None

    extra1 = res1.get('data', {}).get('extraInfo', {})
    if extra1.get('reqHoNumList'):
        return extra1['reqHoNumList']

    return []

def fetch_final_data_step(token, address, dong_code, ho_code):
    """
    [Step 2] 주소 -> 동 -> 호 순서로 호출하여 최종 데이터 반환
    ** 가장 중요: 매 호수마다 이 함수가 처음부터 끝까지 실행되어야 함 **
    """
    # -------------------------------------------------
    # 1차: 주소 검색 (세션 생성)
    # -------------------------------------------------
    res0 = fetch_initial_search(token, address)
    if not res0 or res0['result']['code'] != 'CF-03002':
        return None

    data0 = res0['data']

    # -------------------------------------------------
    # 2차: 동 선택 (dongNum 입력) -> jobIndex 증가됨
    # -------------------------------------------------
    res1 = fetch_next_step(
        token,
        jti=data0['jti'],
        jobIndex=data0['jobIndex'],  # 보통 0
        threadIndex=data0['threadIndex'],
        twoWayTimestamp=data0['twoWayTimestamp'],
        dong_num=dong_code
    )

    if not res1 or res1['result']['code'] != 'CF-03002':
        # 만약 여기서 바로 CF-00000(성공)이 뜨면 호 선택이 필요 없는 경우임
        return res1

    data1 = res1['data']

    # -------------------------------------------------
    # 3차: 호 선택 (hoNum 입력) -> 최종 결과
    # -------------------------------------------------
    # 중요: Step 1의 응답에 있는 jti와 jobIndex를 사용해야 함
    res2 = fetch_next_step(
        token,
        jti=data1['jti'],
        jobIndex=data1['jobIndex'],  # 보통 1로 증가해 있음
        threadIndex=data1['threadIndex'],
        twoWayTimestamp=data1['twoWayTimestamp'],
        dong_num=dong_code,
        ho_num=ho_code  # 여기서 호 정보 입력
    )

    return res2


def select_sample_targets(ho_list):
    """
    전체 호 목록에서 저/중/고층 표본 추출 과정을 출력
    """
    print(f"      [Analyzing] 총 {len(ho_list)}개 호수 층수 분석 시작...")

    # 1. 층별로 호수 그룹핑
    floors_map = {}

    for ho in ho_list:
        ho_name = urllib.parse.unquote_plus(ho['reqHo'])
        floor = extract_floor_from_ho_name(ho_name)

        if floor > 0:  # 지상층만 대상
            if floor not in floors_map:
                floors_map[floor] = []
            floors_map[floor].append(ho)

    if not floors_map:
        print("      [Warning] 분석 가능한 지상층 호수가 없습니다.")
        return []

    # 2. 존재하는 층수 정렬 및 구간 산정
    sorted_floors = sorted(floors_map.keys())
    min_floor = sorted_floors[0]  # 최저층
    max_floor = sorted_floors[-1]  # 최고층
    mid_floor = sorted_floors[len(sorted_floors) // 2]  # 중간층 (인덱스 기준)

    print(f"      [Building Info] 최저 {min_floor}층 ~ 최고 {max_floor}층 (총 {len(sorted_floors)}개 층 존재)")

    # 3. 추출 대상 선정 (중복 제거)
    target_floors = sorted(list(set([min_floor, mid_floor, max_floor])))
    targets = []

    print(f"      [Target Selection] 표본 추출 결과:")

    for f in target_floors:
        # 구간 명칭 결정
        label = "중층"
        if f == min_floor:
            label = "저층(Min)"
        elif f == max_floor:
            label = "고층(Max)"

        # 해당 층의 첫 번째 호수 선택
        # (호수 이름으로 정렬해서 가장 앞 번호 선택, 예: 101호 vs 105호 중 101호)
        floors_map[f].sort(key=lambda x: x['reqHo'])
        target_ho_data = floors_map[f][0]
        target_ho_name = urllib.parse.unquote_plus(target_ho_data['reqHo'])

        targets.append(target_ho_data)

        # 상세 로그 출력
        print(f"         -> [{label}] {f}층: {target_ho_name} 선택됨")

    return targets

# ==========================================
# 5. 메인 로직: 아파트 단지 전체 수집 (Sweeping)
# ==========================================
def collect_apartment_complex(token, start_address):
    """
    단지 내 '중간 동' 선택 로그 및 '저/중/고층' 표본 수집
    """
    print(f"\n===============================================================")
    print(f"[Start] 표본 수집 시작: {start_address}")
    print(f"===============================================================")
    building_name = get_building_name_from_kakao(start_address)
    target_address = f"{start_address} {building_name}"
    # 1. 동(Dong) 목록 확보
    dong_list, res_data = get_dong_list_step(token, target_address)

    if not dong_list:
        # res_data가 있으면 API 코드를, 없으면 일반 에러 메시지 저장
        error_code = "UNKNOWN_ERROR"
        if res_data and 'result' in res_data:
            error_code = res_data['result']['code']
            msg = res_data['result']['message']
            print(f"   [Fail] 동 목록 없음 ({error_code}): {msg}")
        else:
            print("   [Fail] 동 목록 없음 (응답 없음)")

        # 실패 로그 저장 (다음에 조회 안 되게 함)
        save_job_log(start_address, status=error_code)
        return

    # ---------------------------------------------------------
    # [전략 1] 중간 동(Middle Dong) 선택 및 로그
    # ---------------------------------------------------------
    # 동 이름 정렬
    dong_list.sort(key=lambda x: x['reqDong'])

    # 동 목록 간략 출력
    simple_dong_names = [urllib.parse.unquote_plus(d['reqDong']) for d in dong_list]
    print(f"   [Dong List] 발견된 동({len(dong_list)}개): {simple_dong_names}")

    total_dongs = len(dong_list)
    mid_idx = total_dongs // 2

    target_dong = dong_list[mid_idx]  # 중간 동 선택

    dong_name = urllib.parse.unquote_plus(target_dong['reqDong'])
    dong_code = target_dong['commDongNum']

    print(f"   [Selected Dong] 표본 동 선택: '{dong_name}' (리스트의 {mid_idx + 1}번째)")

    # 2. 해당 동의 호(Ho) 목록 확보
    ho_list = get_ho_list_step(token, target_address, dong_code)

    if not ho_list:
        print(f"      [Fail] 호 목록을 가져올 수 없습니다.")
        return

    # ---------------------------------------------------------
    # [전략 2] 저/중/고층 표본 추출 로직
    # ---------------------------------------------------------
    target_hos = select_sample_targets(ho_list)

    print(f"      [Request] API 요청 시작 (총 {len(target_hos)}건)...")

    # 3. 표본 호수 데이터 수집 (API 호출)
    for ho in target_hos:
        ho_name = urllib.parse.unquote_plus(ho['reqHo'])
        ho_code = ho['commHoNum']

        # [핵심] 실제 API 호출
        final_res = fetch_final_data_step(token, target_address, dong_code, ho_code)

        if final_res and final_res['result']['code'] == 'CF-00000':
            # DB 저장 함수 호출 (기존 코드 사용)
            parse_and_save(final_res, dong_name, ho_name)
        else:
            msg = final_res['result']['message'] if final_res else "응답없음"
            print(f"         [Error] {ho_name} 수집 실패: {msg}")

        # API 과부하 방지 딜레이
        time.sleep(0.5)

    print(f"   [Complete] '{dong_name}' 표본 수집 완료.\n")

    # 수집 완료 로그 저장 (그래야 다음 실행 때 건너뜀)
    save_job_log(start_address)
    time.sleep(1)

def save_job_log(address, status="DETAIL_SAVED"):
    """수집 완료 기록 저장"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # 이미 있으면 update, 없으면 insert
        cur.execute("""
            INSERT INTO api_job_log (search_address, status, created_at, updated_at) 
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(search_address) DO UPDATE SET
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
        """, (address, status))
        conn.commit()
    except Exception as e:
        print(f"로그 저장 실패: {e}")
    finally:
        conn.close()

def get_targets_from_rent_db(limit=100):
    """
    [DB 조회] 전월세/실거래가 테이블(raw_rent)에서
    아직 수집하지 않은 건물의 주소를 추출합니다.
    """
    conn = get_connection()
    cur=conn.cursor()

    print("DB에서 실거래가 기반 수집 대상 추출 중...")

    # ---------------------------------------------------------
    # 쿼리 설명:
    # 1. raw_rent(실거래)와 meta_bjdong_codes(법정동명)를 조인
    # 2. api_job_log(작업내역)에 없는(성공하지 않은) 주소만 필터링 (LEFT JOIN ... IS NULL)
    # 3. DISTINCT로 중복 제거
    # ---------------------------------------------------------
    query = """
        SELECT DISTINCT 
            m.bjdong_name, 
            r.본번, 
            r.부번
        FROM raw_rent r
        JOIN meta_bjdong_codes m 
          ON r.시군구 = m.sgg_code AND r.법정동 = m.bjdong_code
        LEFT JOIN api_job_log log 
          ON (
              m.bjdong_name || ' ' || CAST(r.본번 AS INTEGER) || 
              CASE 
                  WHEN CAST(r.부번 AS INTEGER) = 0 THEN '' 
                  ELSE '-' || CAST(r.부번 AS INTEGER) 
              END
          ) = log.search_address
        WHERE log.search_address IS NULL
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


# ==========================================
# 6. 실행부
# ==========================================
if __name__ == "__main__":
    # DB 테이블 생성 (collective_units 테이블 확인)
    init_db()

    token = get_access_token()

    if token:
        target_list = get_targets_from_rent_db(limit=100)

        if not target_list:
            print("모든 데이터가 최신이거나, 수집할 대상이 없습니다.")

        for idx, target_addr in enumerate(target_list):
            print(f"\n===============================================================")
            print(f"[진행률 {idx + 1}/{len(target_list)}] Target: {target_addr}")
            print(f"===============================================================")

            # 3. 아파트 단지 전체 수집 (Sweeping) 실행
            collect_apartment_complex(token, target_addr)

            # 건물이 바뀔 때마다 잠시 휴식 (API 보호)
            time.sleep(2)

    else:
        print("토큰 발급 실패. 설정을 확인하세요.")