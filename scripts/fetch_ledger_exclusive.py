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
    from scripts.kakao_localmap_api import get_building_name_from_kakao
except ModuleNotFoundError:
    # 2. 이 파일을 직접 실행할 때 (현재 폴더 기준)
    from db_manager import init_db, get_connection
    from kakao_localmap_api import get_building_name_from_kakao

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'local_fraud_db.sqlite'))

# ==========================================
# 1. 설정 (Configuration) - 다중 계정 지원
# ==========================================
# 계정 목록 (환경변수에 _1, _2 접미사 붙은 키 필요)
ACCOUNTS = [
    # {
    #     "client_id": os.getenv("CLIENT_ID_1"),
    #     "client_secret": os.getenv("CLIENT_SECRET_1"),
    #     "user_id": os.getenv("CODEF_USER_ID_1"),
    #     "rsa_pass": os.getenv("CODEF_USER_RSA_PASSWORD_1")
    # },
    {
        "client_id": os.getenv("CLIENT_ID_2"),
        "client_secret": os.getenv("CLIENT_SECRET_2"),
        "user_id": os.getenv("CODEF_USER_ID_2"),
        "rsa_pass": os.getenv("CODEF_USER_RSA_PASSWORD_2")
    }
]

CURRENT_ACCOUNT_IDX = 0
CURRENT_TOKEN = None

# API 엔드포인트
TOKEN_URL = "https://oauth.codef.io/oauth/token"
API_URL = "https://development.codef.io/v1/kr/public/lt/eais/aggregate-buildings"


# ==========================================
# 2. 토큰 및 계정 관리 함수
# ==========================================
def get_access_token():
    """현재 활성화된 계정으로 토큰 발급"""
    global CURRENT_ACCOUNT_IDX
    account = ACCOUNTS[CURRENT_ACCOUNT_IDX]

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "scope": "read"}

    try:
        response = requests.post(
            TOKEN_URL,
            headers=headers,
            data=data,
            auth=(account['client_id'], account['client_secret'])
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        print(f"[Account {CURRENT_ACCOUNT_IDX + 1}] 토큰 발급 성공")
        return token
    except Exception as e:
        print(f"[Account {CURRENT_ACCOUNT_IDX + 1}] 토큰 발급 실패: {e}")
        return None


def switch_account():
    """다음 계정으로 전환"""
    global CURRENT_ACCOUNT_IDX, CURRENT_TOKEN

    NEXT_IDX = (CURRENT_ACCOUNT_IDX + 1) % len(ACCOUNTS)
    if NEXT_IDX == 0 and len(ACCOUNTS) > 1:
        print("모든 계정의 한도가 소진되었습니다.")
        sys.exit(0)

    print(f"\n[Switch] 계정 전환: {CURRENT_ACCOUNT_IDX + 1}번 -> {NEXT_IDX + 1}번")
    CURRENT_ACCOUNT_IDX = NEXT_IDX
    CURRENT_TOKEN = get_access_token()
    return CURRENT_TOKEN


def send_api_request(token, payload):
    """API 요청 전송 및 한도 초과 시 계정 전환 로직"""
    global CURRENT_TOKEN
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()

        try:
            result = resp.json()
        except:
            decoded = urllib.parse.unquote_plus(resp.text)
            result = json.loads(decoded)

        # Failover 로직
        if result['result']['code'] == 'CF-00012':
            print(f"[Limit Reached] 계정 {CURRENT_ACCOUNT_IDX + 1} 한도 초과! 전환 시도...")
            new_token = switch_account()

            if new_token:
                # Payload ID/PW 교체
                current_acc = ACCOUNTS[CURRENT_ACCOUNT_IDX]
                if 'id' in payload:
                    payload['id'] = current_acc['user_id']
                    payload['password'] = current_acc['rsa_pass']

                print("[Retry] 새 계정으로 요청 재시도...")
                return send_api_request(new_token, payload)
            else:
                sys.exit(0)

        return result

    except Exception as e:
        print(f"API 요청 실패: {e}")
        return None


# ==========================================
# 3. Codef API 호출 함수들 (수정됨)
# ==========================================
def fetch_initial_search(token, address):
    """[Step 1] 최초 주소 검색"""
    account = ACCOUNTS[CURRENT_ACCOUNT_IDX]
    payload = {
        "organization": "0008",
        "loginType": "1",
        "id": account['user_id'],
        "password": account['rsa_pass'],
        "address": address,
        "dong": ""
    }
    return send_api_request(token, payload)


def fetch_next_step(token, jti, jobIndex, twoWayTimestamp, threadIndex, dong_num=None, ho_num=None):
    """[Step 2 & 3] 2-Way 추가 인증"""
    account = ACCOUNTS[CURRENT_ACCOUNT_IDX]
    payload = {
        "organization": "0008",
        "loginType": "1",
        "id": account['user_id'],
        "password": account['rsa_pass'],
        "is2Way": True,
        "twoWayInfo": {
            "jobIndex": jobIndex,
            "threadIndex": threadIndex,
            "jti": jti,
            "twoWayTimestamp": twoWayTimestamp
        }
    }
    if dong_num: payload["dongNum"] = dong_num
    if ho_num: payload["hoNum"] = ho_num

    return send_api_request(token, payload)


# ==========================================
# 4. 헬퍼 함수
# ==========================================
def extract_floor_from_ho_name(ho_name):
    numbers = re.findall(r'\d+', ho_name)
    if not numbers: return -1
    val = int(numbers[0])
    floor = val // 100
    if 'B' in ho_name or '지' in ho_name or floor == 0: return -1
    return floor


def select_sample_targets(ho_list):
    """저/중/고층 표본 추출"""
    print(f"      [Analyzing] 총 {len(ho_list)}개 호수 층수 분석 시작...")
    floors_map = {}
    for ho in ho_list:
        ho_name = urllib.parse.unquote_plus(ho['reqHo'])
        floor = extract_floor_from_ho_name(ho_name)
        if floor > 0:
            if floor not in floors_map: floors_map[floor] = []
            floors_map[floor].append(ho)

    if not floors_map:
        print("      [Warning] 분석 가능한 지상층 호수가 없습니다.")
        return []

    sorted_floors = sorted(floors_map.keys())
    min_floor, max_floor = sorted_floors[0], sorted_floors[-1]
    mid_floor = sorted_floors[len(sorted_floors) // 2]

    target_floors = sorted(list(set([min_floor, mid_floor, max_floor])))
    targets = []

    for f in target_floors:
        floors_map[f].sort(key=lambda x: x['reqHo'])
        targets.append(floors_map[f][0])

    return targets


# ==========================================
# 5. 데이터 파싱 및 DB 저장 (최종 수정)
# ==========================================
def parse_and_save(api_json, dong_name, ho_name):
    data = api_json.get('data', {})
    if not data: return

    # 1. 건물 기본 정보
    origin_unique_no = data.get('commUniqeNo', '')
    unique_no = f"{origin_unique_no}-{ho_name}" if origin_unique_no else None
    building_id_code = data.get('resDocNo')
    road_addr = urllib.parse.unquote_plus(data.get('commAddrRoadName', '') or '')
    lot_addr = data.get('commAddrLotNumber', '') or ''
    detail_addr = f"{dong_name} {ho_name}"
    is_violating = 'Y' if data.get('resViolationStatus') else 'N'

    # [핵심] resType='0'(전유부) 찾기
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

    # 구조 백업 로직
    if not structure_type or structure_type == "알수없음":
        for item in data.get('resOwnedList', []):
            temp_str = urllib.parse.unquote_plus(item.get('resStructure', '') or '')
            if temp_str:
                structure_type = temp_str
                break

    # 2. 소유자 정보
    owner_nm = ""
    ownership_date = None
    ownership_cause = ""

    if data.get('resOwnerList'):
        current_owner = data['resOwnerList'][0]
        owner_nm = urllib.parse.unquote_plus(current_owner.get('resOwner', '') or '')
        raw_date = current_owner.get('resChangeDate', '')
        if len(raw_date) == 8:
            ownership_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        ownership_cause = urllib.parse.unquote_plus(current_owner.get('resChangeReason', '') or '')

    # 3. DB 저장
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id FROM building_info WHERE road_address = ? AND detail_address = ?
        """, (road_addr, detail_addr))
        row = cur.fetchone()

        if row:
            building_id = row[0]
            cur.execute("""
                UPDATE building_info 
                SET unique_number=?, building_id_code=?, exclusive_area=?, 
                    main_use=?, structure_type=?, owner_name=?, 
                    ownership_changed_date=?, ownership_cause=?, is_violating_building=?
                WHERE id = ?
            """, (unique_no, building_id_code, exclusive_area, main_use, structure_type,
                  owner_nm, ownership_date, ownership_cause, is_violating, building_id))
        else:
            cur.execute("""
                INSERT INTO building_info (
                    unique_number, building_id_code, road_address, lot_address, detail_address,
                    exclusive_area, main_use, structure_type, owner_name, 
                    ownership_changed_date, ownership_cause, is_violating_building
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (unique_no, building_id_code, road_addr, lot_addr, detail_addr,
                  exclusive_area, main_use, structure_type, owner_nm,
                  ownership_date, ownership_cause, is_violating))
            building_id = cur.lastrowid

        # 가격 정보 갱신
        if building_id:
            cur.execute("DELETE FROM public_price_history WHERE building_info_id = ?", (building_id,))
            price_list = data.get('resPriceList', [])
            insert_data = []
            for price_item in price_list:
                raw_price = price_item.get('resBasePrice', '0').replace(',', '')
                raw_date = price_item.get('resReferenceDate', '')
                fmt_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if len(raw_date) == 8 else raw_date
                insert_data.append((building_id, fmt_date, raw_price))

            if insert_data:
                cur.executemany(
                    "INSERT INTO public_price_history (building_info_id, base_date, price) VALUES (?, ?, ?)",
                    insert_data)

            conn.commit()
            print(f"      [Saved] {dong_name} {ho_name} (ID: {building_id}, 주인: {owner_nm})")

    except Exception as e:
        conn.rollback()
        print(f"      [Error] 저장 실패: {e}")
    finally:
        conn.close()


# ==========================================
# 6. 단계별 API 호출 로직 (수정됨)
# ==========================================
def get_dong_list_step(token, address):
    """[Step 0] 주소 검색 및 동 목록 확보"""
    res = fetch_initial_search(token, address)
    if not res: return None, None

    code = res['result']['code']
    data = res.get('data', {})
    extra = data.get('extraInfo', {})

    if extra.get('reqDongNumList'):
        return extra['reqDongNumList'], data
    elif extra.get('reqHoNumList'):
        # 단일 건물일 경우 가상의 동 생성 (commDongNum 비움)
        return [{'reqDong': '단일건물', 'commDongNum': ''}], data
    elif code == 'CF-00000':
        return [], data

    return [], res


def get_ho_list_step(token, address, dong_code):
    """
    호 목록 반환 (API 응답 코드 검사 추가)
    """
    # 1. 주소 검색 (세션 시작)
    res0 = fetch_initial_search(token, address)
    if not res0: return None

    code = res0['result']['code']
    msg = res0['result']['message']
    data0 = res0.get('data', {})
    extra0 = data0.get('extraInfo', {})

    # [예외 처리] API 오류이거나, 세션 키(jti)가 없는 경우 중단
    if code != 'CF-03002' and code != 'CF-00000':
        print(f"      [Warning] 호 목록 조회 진입 실패 ({code}): {msg}")
        return []

    # [핵심] 동 코드가 없으면(단일 건물), 1차 결과의 호 목록 바로 반환
    if not dong_code:
        if extra0.get('reqHoNumList'): return extra0['reqHoNumList']
        if data0.get('resHoList'): return data0['resHoList']
        return []

    # -------------------------------------------------------
    # 동 코드가 있는 경우 (아파트) -> 동 선택 요청 수행
    # -------------------------------------------------------

    # [Safety Check] 동 선택을 하려면 jti가 필수인데, 없는 경우 방어
    if 'jti' not in data0:
        # CF-00000 등으로 바로 끝나버려서 jti가 없는 경우 등
        return []

    # 2. 동 선택 요청
    res1 = fetch_next_step(
        token, data0['jti'], data0['jobIndex'], data0['twoWayTimestamp'], data0['threadIndex'],
        dong_num=dong_code
    )

    if not res1: return None

    # 결과 코드 확인 (여기서도 실패할 수 있음)
    if res1['result']['code'] != 'CF-03002':
        return []

    extra1 = res1.get('data', {}).get('extraInfo', {})
    if extra1.get('reqHoNumList'): return extra1['reqHoNumList']

    return []

def fetch_final_data_step(token, address, dong_code, ho_code):
    """
    [Step 2] 최종 데이터 조회 (동 코드 유무에 따라 분기)
    """
    # 1차: 주소 검색 (세션 생성)
    res0 = fetch_initial_search(token, address)
    if not res0: return None

    # [수정] 결과 코드 검사 (jti 존재 여부 확인)
    code = res0['result']['code']
    data0 = res0.get('data', {})

    # CF-03002가 아니면 jti가 없으므로 진행 불가
    # (단일 건물인데 호 목록이 바로 나오는 경우는 이미 앞단 get_ho_list_step에서 처리했으므로 여기선 무조건 jti가 있어야 함)
    if code != 'CF-03002' or 'jti' not in data0:
        msg = res0['result']['message']
        print(f"         [Error] 상세 조회 진입 실패 ({code}): {msg}")
        return None

    # Case A: 단일 건물 -> 바로 호 선택
    if not dong_code:
        return fetch_next_step(
            token, data0['jti'], data0['jobIndex'], data0['twoWayTimestamp'], data0['threadIndex'],
            ho_num=ho_code
        )

    # Case B: 아파트 -> 동 선택 -> 호 선택
    # 동 선택 요청
    res1 = fetch_next_step(
        token, data0['jti'], data0['jobIndex'], data0['twoWayTimestamp'], data0['threadIndex'],
        dong_num=dong_code
    )

    # 동 선택 실패 시 (jti 없음 등)
    if not res1 or res1['result']['code'] != 'CF-03002':
        return None

    data1 = res1.get('data', {})

    return fetch_next_step(
        token, data1['jti'], data1['jobIndex'], data1['twoWayTimestamp'], data1['threadIndex'],
        dong_num=dong_code, ho_num=ho_code
    )


# ==========================================
# 7. 메인 실행 로직
# ==========================================
def collect_apartment_complex(token, start_address):
    print(f"\n===============================================================")
    print(f"[Start] 표본 수집 시작: {start_address}")
    print(f"===============================================================")

    # 주소 변환 (Kakao API 사용)
    building_name = get_building_name_from_kakao(start_address)
    target_address = f"{start_address} {building_name}" if building_name else start_address

    # 1. 동 목록 확보
    dong_list, res_data = get_dong_list_step(token, target_address)

    if not dong_list:
        print("   [Fail] 동 목록 없음")
        save_job_log(start_address, status="DATA_NOT_FOUND")
        return

    # [중간 동 선택]
    valid_dongs = []
    skip_keywords = ['상가', '근린', '경비실', '주차장', '기계실', '관리동', '노인정', '유치원']

    for d in dong_list:
        d_name = urllib.parse.unquote_plus(d.get('reqDong', '')).strip()
        if any(k in d_name for k in skip_keywords): continue
        valid_dongs.append(d)

    if not valid_dongs:
        print("   [Skip] 수집할 주거용 동이 없습니다.")
        save_job_log(start_address, status="NO_RESIDENTIAL_DONG")
        return

    valid_dongs.sort(key=lambda x: x.get('reqDong'))
    target_dong = valid_dongs[len(valid_dongs) // 2]

    target_dong_name = urllib.parse.unquote_plus(target_dong.get('reqDong', '')).strip()
    target_dong_code = target_dong.get('commDongNum')  # 단일 건물이면 ''

    print(f"   [Selected Dong] '{target_dong_name}' (Code: {target_dong_code})")

    # 2. 호 목록 확보
    ho_list = get_ho_list_step(token, target_address, target_dong_code)

    if not ho_list:
        print(f"      [Fail] 호 목록 없음")
        return

    # 3. 표본 수집 & 저장
    target_hos = select_sample_targets(ho_list)
    print(f"      [Request] 표본 {len(target_hos)}건 수집 시작...")

    for ho in target_hos:
        ho_name = urllib.parse.unquote_plus(ho['reqHo'])
        ho_code = ho['commHoNum']

        final_res = fetch_final_data_step(token, target_address, target_dong_code, ho_code)

        if final_res and final_res['result']['code'] == 'CF-00000':
            parse_and_save(final_res, target_dong_name, ho_name)
        else:
            msg = final_res['result']['message'] if final_res else "응답없음"
            print(f"         [Error] {ho_name} 수집 실패: {msg}")

        time.sleep(0.3)

    print(f"   [Complete] '{target_dong_name}' 수집 완료.\n")
    save_job_log(start_address, status="DETAIL_SAVED")
    time.sleep(1)


def save_job_log(address, status="DETAIL_SAVED"):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO api_job_log (search_address, job_type, status, created_at, updated_at) 
            VALUES (?, 'EXCLUSIVE', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(search_address, job_type) DO UPDATE SET
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
        """, (address, status))
        conn.commit()
    except Exception as e:
        print(f"로그 저장 실패: {e}")
    finally:
        conn.close()


def get_targets_from_rent_db(limit=100):
    conn = get_connection()
    cur = conn.cursor()

    # 아파트/오피스텔/연립다세대 랜덤 수집 쿼리 (아파트는 임시로 제외-> 데이터 불균형성 해소 위함)
    query = """
        SELECT * FROM (
            SELECT DISTINCT m.bjdong_name, r.본번, r.부번
            FROM raw_rent r
            JOIN meta_bjdong_codes m ON r.시군구 = m.sgg_code AND r.법정동 = m.bjdong_code
            LEFT JOIN api_job_log log 
              ON (m.bjdong_name || ' ' || CAST(r.본번 AS INTEGER) || CASE WHEN CAST(r.부번 AS INTEGER) = 0 THEN '' ELSE '-' || CAST(r.부번 AS INTEGER) END) = log.search_address
              AND log.job_type = 'EXCLUSIVE'
            WHERE log.search_address IS NULL AND r.건물유형 = '연립다세대'
            ORDER BY RANDOM() LIMIT 20
        )
        UNION ALL
        SELECT * FROM (
            SELECT DISTINCT m.bjdong_name, r.본번, r.부번
            FROM raw_rent r
            JOIN meta_bjdong_codes m ON r.시군구 = m.sgg_code AND r.법정동 = m.bjdong_code
            LEFT JOIN api_job_log log 
              ON (m.bjdong_name || ' ' || CAST(r.본번 AS INTEGER) || CASE WHEN CAST(r.부번 AS INTEGER) = 0 THEN '' ELSE '-' || CAST(r.부번 AS INTEGER) END) = log.search_address
              AND log.job_type = 'EXCLUSIVE'
            WHERE log.search_address IS NULL AND r.건물유형 = '오피스텔'
            ORDER BY RANDOM() LIMIT 20
        )
    """
    try:
        cur.execute(query)
        rows = cur.fetchall()
        address_list = []
        for row in rows:
            bjdong_name = row[0]
            bonbeon = int(row[1])
            bubeon = int(row[2])
            if bubeon == 0:
                addr = f"{bjdong_name} {bonbeon}"
            else:
                addr = f"{bjdong_name} {bonbeon}-{bubeon}"
            address_list.append(addr)
        print(f"수집 대상 {len(address_list)}건 확보")
        return address_list
    except Exception as e:
        print(f"DB 조회 실패: {e}")
        return []
    finally:
        conn.close()


if __name__ == "__main__":
    CURRENT_TOKEN = get_access_token()
    if CURRENT_TOKEN:
        target_list = get_targets_from_rent_db(limit=60)
        if not target_list:
            print("수집할 대상이 없습니다.")
        else:
            for idx, target_addr in enumerate(target_list):
                print(f"\n[{idx + 1}/{len(target_list)}] Target: {target_addr}")
                collect_apartment_complex(CURRENT_TOKEN, target_addr)
                time.sleep(1)