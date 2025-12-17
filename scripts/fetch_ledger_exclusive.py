import os, time, sys, re
from dotenv import load_dotenv
import requests
import json
import sqlite3
import urllib.parse
import pandas as pd
# [수정] 경로 문제 해결을 위한 조건부 임포트
try:
    # 1. 외부(predict.py 등)에서 패키지로 불러올 때 (프로젝트 루트 기준)
    from scripts.db_manager import init_db, get_connection
    from scripts.kakao_localmap_api import get_building_name_from_kakao, get_road_address_from_kakao, get_all_address_and_building_from_kakao
except ModuleNotFoundError:
    # 2. 이 파일을 직접 실행할 때 (현재 폴더 기준)
    from db_manager import init_db, get_connection
    from kakao_localmap_api import get_building_name_from_kakao, get_road_address_from_kakao, get_all_address_and_building_from_kakao

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'local_fraud_db.sqlite'))

# ==========================================
# 1. 설정 (Configuration) - 다중 계정 지원
# ==========================================
# 계정 목록 (환경변수에 _1, _2 접미사 붙은 키 필요)
CLIENT_ID = os.getenv("CLIENT_ID_1")
CLIENT_SECRET = os.getenv("CLIENT_SECRET_1")
CODEF_USER_ID = os.getenv("CODEF_USER_ID_1")
CODEF_USER_RSA_PASSWORD = os.getenv("CODEF_USER_RSA_PASSWORD_1")

# API 엔드포인트
TOKEN_URL = "https://oauth.codef.io/oauth/token"
API_URL = "https://development.codef.io/v1/kr/public/lt/eais/aggregate-buildings"


# ==========================================
# 2. 토큰 및 계정 관리 함수
# ==========================================
def get_access_token():

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "scope": "read"}

    try:
        response = requests.post(
            TOKEN_URL,
            headers=headers,
            data=data,
            auth=(CLIENT_ID, CLIENT_SECRET)
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        return token
    except Exception as e:
        print(f"[Error] {e}")
        return None


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
            print(f"CF-00012 일일 100회 제한이 초과되었습니다.")
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
    payload = {
        "organization": "0008",
        "loginType": "1",
        "id": CODEF_USER_ID,
        "password": CODEF_USER_RSA_PASSWORD,
        "address": address,
        "dong": ""
    }
    return send_api_request(token, payload)


def fetch_next_step(token, jti, jobIndex, twoWayTimestamp, threadIndex, dong_num=None, ho_num=None):
    """[Step 2 & 3] 2-Way 추가 인증"""
    payload = {
        "organization": "0008",
        "loginType": "1",
        "id": CODEF_USER_ID,
        "password": CODEF_USER_RSA_PASSWORD,
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
        if extra0.get('reqHoNumList'):
            return extra0['reqHoNumList']
        if data0.get('resHoList'):
            return data0['resHoList']
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
    if extra1.get('reqHoNumList'):
        return extra1['reqHoNumList']

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

    # 1. 동 목록 확보 (지번 + 건물명)
    dong_list, res_data = get_dong_list_step(token, target_address)

    # 2차 시도: 1차 실패 시 실행 (도로명 + 건물명)
    if not dong_list:
        print(f"   [Retry] '{target_address}' 조회 실패 -> 도로명 주소로 재시도")

        try:
            # 도로명 주소 변환 (여기서만 Kakao API 추가 호출)
            road_part = get_road_address_from_kakao(start_address)

            if road_part:
                # 아까 구해둔 building_name 재사용 (API 호출 절약)
                retry_address = f"{road_part} {building_name}".strip()

                print(f"   [Retry] 재시도 주소: {retry_address}")
                dong_list, res_data = get_dong_list_step(token, retry_address)
            else:
                print("   [Retry Fail] 도로명 주소를 찾을 수 없음")

        except Exception as e:
            # Kakao API 장애나 파싱 에러 등으로 멈추지 않도록 처리
            print(f"   [Error] 재시도 중 주소 변환 오류 발생: {e}")

    # 최종 확인: 그래도 없으면 실패 처리
    if not dong_list:
        print("   [Fail] 동 목록 없음 (최종)")
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

def _collect_exclusive_with_retry(token, address):
    """
    중간 호/층 조회
    """
    print(f"      [Work] 전유부(Exclusive) 수집을 시작합니다.")
    # 1차 시도: 입력받은 지번 주소로 시도
    jibun_part, road_part, build_part = get_all_address_and_building_from_kakao(address)
    address = f"{jibun_part} {build_part}".strip()
    if fetch_target_middle_unit(token, address):
        print(f"      [Success] 지번 전유부(Exclusive) 수집 성공 {address}")
        return True

    try:
        print(f"      [Fail] 지번 전유부(Exclusive) 수집 실패")
        print(f"      [Work] 도로명으로 전유부(Exclusive) 수집을 재시도합니다.")
        retry_address = f"{road_part} {build_part}".strip()

        if fetch_target_middle_unit(token, retry_address):
            print(f"      [Success] 도로명 전유부(Exclusive) 수집 성공 {retry_address}")
            return True
    except Exception as e:
        print(f"      [Error] 전유부 재시도 주소 생성 실패: {e}")

    return False

# 각 지역 시군구별 아파트/오피스텔/연립다세대 1건씩 조회 (총 9건)
def get_targets_from_rent_db(token, search_sgg):
    conn = get_connection()
    cur = conn.cursor()

    try:
        # [Step 1] 작업할 대상 1개 가져오기 (LRU 방식)
        # SQLite에서는 ORDER BY ASC 시 NULL(한 번도 안 한 것)이 자동으로 맨 먼저 옵니다.
        select_sql = """
            SELECT sgg_code 
            FROM job_sgg_history
            WHERE sgg_code= ?
            ORDER BY last_worked_at ASC
            LIMIT 1
        """
        cur.execute(select_sql, (search_sgg,))
        row = cur.fetchone()
        if not row:
            print("  [Wait] 작업할 대상이 없습니다. 대기 중...")
            return False  # 대기 신호

        target_sgg = row[0]

        # [Step 2] 상태 'DOING'으로 변경 (Locking)
        cur.execute("UPDATE job_sgg_history SET status='DOING' WHERE sgg_code=?", (target_sgg,))
        conn.commit()

        print(f"[{target_sgg}] 데이터 수집 시작...")

        # [Step 3] 실제 랜덤 데이터 조회 (SQLite 문법 적용)
        # 문자열 연결: || 사용
        # 랜덤 함수: RANDOM()
        data_sql = """
            SELECT * FROM (
                SELECT 
                    m.bjdong_name, 
                    r.본번, 
                    r.부번, 
                    r.건물유형,
                    ROW_NUMBER() OVER (PARTITION BY r.건물유형 ORDER BY RANDOM()) as rn
                FROM raw_rent r
                JOIN meta_bjdong_codes m ON r.시군구 = m.sgg_code AND r.법정동 = m.bjdong_code
                LEFT JOIN api_job_log log 
                    ON (m.bjdong_name || ' ' || CAST(r.본번 AS INTEGER) || 
                        CASE WHEN CAST(r.부번 AS INTEGER) = 0 THEN '' ELSE '-' || CAST(r.부번 AS INTEGER) END) 
                        = log.search_address
                    AND log.job_type = 'EXCLUSIVE'
                LEFT JOIN building_info bi
                    ON r.시군구=SUBSTR(bi.unique_number, 1, 5)
                    AND r.법정동=SUBSTR(bi.unique_number, 6, 5)
                    AND r.본번 = SUBSTR(bi.unique_number, 14, 4)
                    AND r.부번 = SUBSTR(bi.unique_number, 18, 4)
                WHERE 
                    r.시군구 = ? 
                    AND r.건물유형 IN ('아파트', '오피스텔', '연립다세대')
                    AND log.search_address IS NULL
                    AND bi.unique_number IS NULL
            ) sub
            WHERE rn <= 1
        """

        # pandas read_sql 실행 (params는 튜플/리스트 형태)
        df_result = pd.read_sql(data_sql, conn, params=(target_sgg,))

        def make_address(row):
            """
            서울특별시 성동구 마장동  0339  0011 를
            서울특별시 성동구 마장동 339-11 로 변환하는 함수
            """
            # 1. 문자를 숫자로 변환하여 앞의 0 제거 ('0128' -> 128)
            main_num = int(row['본번'])
            sub_num = int(row['부번'])

            # 2. 지번 생성 (부번이 0이면 본번만, 아니면 본번-부번)
            if sub_num == 0:
                jibun = f"{main_num}"
            else:
                jibun = f"{main_num}-{sub_num}"

            # 3. 전체 주소 결합
            return f"{row['bjdong_name']} {jibun}"

        # DataFrame에 새로운 컬럼으로 추가
        df_result['full_address'] = df_result.apply(make_address, axis=1)

        # 결과 출력 (리스트로 변환하고 싶다면 .tolist() 사용)
        target_addrs = df_result['full_address'].tolist()

        # API 호출
        for target_addr in target_addrs:
            if token:
                # 중간 동, 중간 호만을 처리
                _collect_exclusive_with_retry(token, target_addr)
            else:
                print("토큰이 없습니다.")
                return False

        # [Step 4] 작업 완료 처리 (현재 시간 기록)
        update_sql = """
            UPDATE job_sgg_history
            SET status = 'DONE',
                last_worked_at = DATETIME('now', 'localtime')
            WHERE sgg_code = ?
        """
        cur.execute(update_sql, (target_sgg,))
        conn.commit()

        if not df_result.empty:
            print(f"  -> [{target_sgg}] {len(df_result)}건 처리 및 완료 기록 저장됨.")
        else:
            print(f"  -> [{target_sgg}] 수집 대상 없음 (완료 처리됨).")

        return True  # 작업 성공 신호

    except Exception as e:
        print(f"[Error] {e}")
        conn.rollback()
        # 에러 발생 시 상태를 FAIL로 변경하고 시간 갱신 (그래야 나중에 다시 시도하거나 후순위로 밀림)
        if 'target_sgg' in locals():
            cur.execute("""
                UPDATE job_sgg_history 
                SET status='FAIL', last_worked_at=DATETIME('now', 'localtime'), message=? 
                WHERE sgg_code=?
            """, (str(e), target_sgg))
            conn.commit()
        return False


import re
import urllib.parse


def extract_floor_from_ho_name(ho_name):
    """
    [Helper] 호수 명칭에서 층수 추출
    Case 1: '1층103호', '1층 103호' -> 명시된 '1' 추출
    Case 2: '1504호' -> 100으로 나눈 몫 (15층)
    Case 3: 'B101', '지하' -> 제외 (-1)
    """
    if not ho_name:
        return -1

    # [전처리] 공백 제거 및 지하/B 확인
    clean_name = ho_name.replace(" ", "").upper()

    # 1. 지하나 B가 포함되면 무조건 제외 (비즈니스 로직에 따라 다름)
    if 'B' in clean_name or '지하' in clean_name:
        return -1

    # 2. [우선순위 1] 'N층' 이라고 명시된 패턴 찾기 (예: 1층103호)
    # (\d+)층 : 숫자 뒤에 바로 '층'이 오는 경우
    explicit_floor = re.search(r'(\d+)층', clean_name)
    if explicit_floor:
        return int(explicit_floor.group(1))

    # 3. [우선순위 2] 명시된 층이 없으면, 숫자 파싱 (예: 1504호)
    numbers = re.findall(r'\d+', clean_name)
    if not numbers:
        return -1

    # 호명에서 가장 긴 숫자를 호수라고 가정하거나, 맨 앞 숫자를 사용
    # "1층103호"는 위에서 걸러졌으므로, 여기는 "1504호", "101호" 같은 케이스만 옴
    val = int(numbers[0])

    # 4자리 이상(1001호~) 또는 3자리(101호~) -> 앞자리를 층수로
    floor = val // 100

    if floor == 0:
        return -1  # 0층은 없으므로 제외

    return floor


def fetch_target_middle_unit(token, target_address, original_address=None):
    """
    [수정됨] 특정 주소의 [중간 동] -> [중간 층 호수]를 수집하여 표본의 대표성을 높임.
    성공 시 True, 실패 시 False 반환.
    """
    # 로깅용 주소 (지번이 없으면 타겟 주소 사용)
    addr_log = original_address if original_address else target_address
    print(f"      [Sub-Task] '{addr_log}' 표본(중간값) 수집 시도...")

    # 1. 동 목록 확보
    dong_list, res_data = get_dong_list_step(token, target_address)

    if not dong_list:
        return False

    # ---------------------------------------------------------
    # [Logic Change] 중간 동(Median Dong) 선택
    # ---------------------------------------------------------
    # 가나다/숫자 순 정렬
    dong_list.sort(key=lambda x: x.get('reqDong', ''))

    # 상가, 관리동 등 비주거용 제외 필터링 (선택사항)
    valid_dongs = [d for d in dong_list if '상가' not in d.get('reqDong', '')]
    if not valid_dongs: valid_dongs = dong_list  # 다 걸러졌으면 원본 사용

    # 중간 인덱스 선택
    mid_idx = len(valid_dongs) // 2
    target_dong = valid_dongs[mid_idx]

    target_dong_name = urllib.parse.unquote_plus(target_dong.get('reqDong', '')).strip()
    target_dong_code = target_dong.get('commDongNum')

    print(f"      [Selected Dong] 중간 동 선택: '{target_dong_name}' ({mid_idx + 1}/{len(valid_dongs)})")

    # 2. 호 목록 확보
    ho_list = get_ho_list_step(token, target_address, target_dong_code)

    if not ho_list:
        return False

    # ---------------------------------------------------------
    # [Logic Change] 중간 층(Median Floor)의 호수 선택
    # ---------------------------------------------------------
    # 층별로 호수 그룹화
    floors_map = {}
    for ho in ho_list:
        h_name = urllib.parse.unquote_plus(ho['reqHo'])
        floor = extract_floor_from_ho_name(h_name)

        if floor > 0:  # 지상층만 대상
            if floor not in floors_map: floors_map[floor] = []
            floors_map[floor].append(ho)

    target_ho = None

    if floors_map:
        # 존재하는 층수들을 정렬 (예: 1, 2, ..., 15)
        sorted_floors = sorted(floors_map.keys())

        # 중간 층 선택 (예: 15층 건물이면 7~8층)
        mid_floor_idx = len(sorted_floors) // 2
        mid_floor = sorted_floors[mid_floor_idx]

        # 해당 층의 첫 번째 호수 선택 (예: 801호)
        target_ho = floors_map[mid_floor][0]
        print(f"      [Selected Floor] 총 {len(sorted_floors)}개 층 중 {mid_floor}층 선택")
    else:
        # 층수 파싱 실패 시(빌라 등) 그냥 중간 인덱스 호수 선택
        target_ho = ho_list[len(ho_list) // 2]
        print(f"      [Selected Floor] 층수 파싱 불가 -> 단순 중간 호수 선택")

    ho_name = urllib.parse.unquote_plus(target_ho['reqHo'])
    ho_code = target_ho['commHoNum']

    print(f"      [Selected Ho] 최종 타겟: {ho_name}")

    # 3. 최종 데이터 조회 및 저장
    final_res = fetch_final_data_step(token, target_address, target_dong_code, ho_code)

    if final_res and final_res['result']['code'] == 'CF-00000':
        parse_and_save(final_res, target_dong_name, ho_name)
        return True
    else:
        return False

# 서울, 인천, 경기 지역의 랜덤 순서로 시군구 코드를 반환
def get_random_sgg_codes():
    conn = get_connection()

    sql = """
    SELECT DISTINCT sgg_code
    FROM meta_bjdong_codes
    WHERE bjdong_name LIKE '서울%'
        OR bjdong_name LIKE '인천%'
        OR bjdong_name LIKE '경기%'
    ORDER BY RANDOM()
    """

    df_rows=pd.read_sql(sql,conn)

    result_list = df_rows['sgg_code'].tolist()

    if not result_list:
        print("  [Wait] 작업할 대상이 없습니다. 대기 중...")
        return []

    return result_list

if __name__ == "__main__":
    CURRENT_TOKEN = get_access_token()
    random_sgg_rows=get_random_sgg_codes()

    for target_sgg in random_sgg_rows:
        # 루프별로 시군구 지역별 아파트/오피스텔/연립다세대 각 1개씩 조회
        get_targets_from_rent_db(CURRENT_TOKEN,target_sgg)

