import pandas as pd
import numpy as np
import joblib
import os
import sys
import re
from datetime import datetime, timedelta

# --- 프로젝트 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- 모듈 임포트 ---
from scripts.fetch_ledger_exclusive import (
    fetch_final_data_step, parse_and_save,
    get_dong_list_step, get_ho_list_step,
    get_access_token, fetch_initial_search,  # 토큰 발급용
    fetch_target_middle_unit, save_job_log
)
from scripts.fetch_ledger_title import fetch_step1_search, fetch_step2_detail, parse_and_save_title
from scripts.kakao_localmap_api import get_road_address_from_kakao, get_building_name_from_kakao
from scripts.data_processor import (
    _create_join_key_from_unique_no, _extract_floor_from_detail, engine
)
from scripts.db_manager import get_connection
from scripts.fetch_trade_data import fetch_trade_data_and_save, get_bjdong_code_map
from scripts.fetch_rent_data import fetch_rent_data_and_save

# --- 모델 및 설정 로드 ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'fraud_rf_model.pkl')
try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"모델 로드 성공: {MODEL_PATH}")
except:
    print("모델 파일이 없습니다. 먼저 학습(train_model.py)을 실행하세요.")
    sys.exit(1)

# 중앙 설정 파일(engine) 임포트
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(project_root)
CSV_PATH = os.path.join(project_root, 'data', '국토교통부_법정동코드_20250805.csv')
df_bjd = pd.read_csv(CSV_PATH, sep=',', encoding='cp949', dtype=str)

# 학습 때 사용한 컬럼 순서 (매우 중요! 순서 틀리면 예측 엉망됨)
# train_model.py에서 학습할 때 썼던 features 리스트와 똑같아야 함
MODEL_FEATURES = [
    'jeonse_ratio', 'hug_risk_ratio', 'total_risk_ratio', 'building_age',
    'parking_per_household', 'is_micro_complex', 'estimated_loan_ratio',
    'is_trust_owner', 'short_term_weight'
]
# One-Hot Encoding용 기본 컬럼들
USE_COLS = ['use_아파트', 'use_오피스텔', 'use_다세대주택', 'use_근린생활시설', 'use_기타']

# 2. 법정동 매핑 딕셔너리 생성 (주소 -> 코드)
# 검색 속도를 높이기 위해 딕셔너리로 변환합니다.
bjd_map = dict(zip(df_bjd['법정동명'], df_bjd['법정동코드']))

def convert_address_to_unique_no(address, bjd_mapping):
    """
    주소 문자열을 파싱하여 고유 번호 포맷으로 변환
    입력: "인천광역시 부평구 부평동 65-124"
    출력: "2823710100-3-00650124"
    """

    # 1. 번지(숫자)와 그 앞의 주소(동 명칭) 분리
    # 정규식 설명: 마지막에 나오는 숫자(본번-부번)를 그룹으로 잡음
    match = re.search(r'(.+)\s+(\d+(?:-\d+)?)', address)

    if not match:
        return None, "주소 형식 오류 (번지 없음)"

    region_addr = match.group(1).strip()  # "인천광역시 부평구 부평동"
    bunji_str = match.group(2).strip()  # "65-124"

    # 2. 법정동 코드 조회
    if region_addr not in bjd_mapping:
        return None, f"법정동 코드 없음 ({region_addr})"

    bjd_code = bjd_mapping[region_addr]

    # 3. 본번/부번 패딩 (4자리 0 채움)
    if '-' in bunji_str:
        main_no, sub_no = bunji_str.split('-')
    else:
        main_no = bunji_str
        sub_no = "0"

    formatted_bunji = f"{int(main_no):04d}{int(sub_no):04d}"

    # 4. 필지 구분 (Land Type)
    # 일반적인 PNU: 1(대지), 2(산)
    # 사용자 요청 예시: 3
    land_type = "3"

    # 만약 주소에 '산'이 포함되면 2로 처리해야 한다면 아래 주석 해제
    # if "산" in address: land_type = "2"

    # 5. 최종 조합
    unique_no = f"{bjd_code}-{land_type}-{formatted_bunji}"

    return unique_no, "성공"

def fetch_real_price_from_api(sigungu_code, bjdong_code, bonbeon, bubeon):
    """
    [신규] 실거래가(매매/전세) API를 호출하여 DB를 최신화하는 함수
    """
    # 1. 조회할 기간 설정 (최근 3개월치 조회)
    # 실거래가는 신고 기한(30일)이 있으므로 최근 데이터가 없을 수 있음
    from datetime import datetime, timedelta

    # 법정동 코드 맵 로드 (API 호출에 필요)
    code_map = get_bjdong_code_map()

    # 최근 3개월 루프
    for i in range(3):
        target_date = datetime.now() - timedelta(days=30 * i)
        deal_ymd = target_date.strftime('%Y%m')

        print(f"   [System] 실거래가 데이터 조회 중 ({deal_ymd})...")

        # 매매 데이터 수집
        fetch_trade_data_and_save(sigungu_code, deal_ymd, code_map)

        # 전세 데이터 수집
        fetch_rent_data_and_save(sigungu_code, deal_ymd, code_map)

    # 수집 후에는 DB에 저장되므로 별도 리턴값 없이 종료
    return True


# =========================================================
# [Helper] 전유부/표제부 수집 상태 확인 함수
# =========================================================
def check_data_existence_by_pnu(table_name, unique_number_prefix):
    """
    [수정됨] 고유번호(PNU) 접두어를 받아 실제 데이터(building_info) 존재 여부 확인
    입력: '2823710100-3-00650124' (특정 지번)
    쿼리: '2823710100-3-00650124%' (해당 지번의 모든 호수 검색)
    """
    if not unique_number_prefix:
        return False

    conn = get_connection()
    cur = conn.cursor()

    try:
        # api_job_log(로그)가 아닌 building_info(실제 데이터)를 직접 확인
        query = f"""
                SELECT 1 FROM {table_name} 
                WHERE unique_number LIKE ? 
                LIMIT 1
            """
        # PNU 뒤에 %를 붙여서 해당 건물의 어떠한 호수라도 있는지 확인
        cur.execute(query, (unique_number_prefix + '%',))
        result = cur.fetchone()
        return result is not None
    except Exception as e:
        print(f"      [Check Error] 데이터 확인 중 오류: {e}")
        return False
    finally:
        conn.close()


def _collect_exclusive_with_retry(token, address):
    """
    [Internal] 전유부 수집 실행 (지번 시도 -> 실패시 도로명 재시도)
    성공 시 True, 실패 시 False 반환
    """
    print(f"      [Work] 전유부(Exclusive) 수집 시작...")

    # 1차 시도: 입력받은 지번 주소로 시도
    if fetch_target_middle_unit(token, address, address):
        return True

    # 2차 시도: 도로명 주소 + 건물명 조합으로 재시도
    try:
        road_part = get_road_address_from_kakao(address)
        build_part = get_building_name_from_kakao(address)
        retry_address = f"{road_part} {build_part}".strip()

        print(f"      [Retry] 전유부: 번지 실패 -> 도로명 재시도: {retry_address}")
        if fetch_target_middle_unit(token, retry_address, address):
            return True
    except Exception as e:
        print(f"      [Error] 전유부 재시도 주소 생성 실패: {e}")

    return False


def _collect_title_with_retry(token, address):
    """
    [Internal] 표제부 수집 실행 (지번 시도 -> 실패시 도로명 재시도)
    성공 시 True, 실패 시 False 반환
    """
    print(f"      [Work] 표제부(Title) 수집 시작...")

    # 순환 참조 방지를 위해 함수 내부 import
    from scripts.fetch_ledger_title import collect_title_data

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

def check_price_log(sigungu_code, deal_ymd, data_type):
    """
    해당 지역/년월/타입의 데이터가 이미 수집되었는지 확인
    (하루가 지나면 재수집하도록 로직 구성 가능)
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        # 오늘 수집한 기록이 있는지 확인 (date(collected_at) = date('now', 'localtime'))
        # 혹은 그냥 존재하는지만 확인해도 됨
        query = """
            SELECT 1 FROM api_price_log 
            WHERE sigungu_code = ? 
              AND deal_ymd = ? 
              AND data_type = ?
              AND substr(collected_at, 1, 10) = date('now')
        """
        cur.execute(query, (sigungu_code, deal_ymd, data_type))
        return cur.fetchone() is not None
    finally:
        conn.close()

def update_price_log(sigungu_code, deal_ymd, data_type):
    """
    수집 완료 기록 저장 (데이터가 0건이어도 기록함)
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        query = """
            INSERT INTO api_price_log (sigungu_code, deal_ymd, data_type, collected_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(sigungu_code, deal_ymd, data_type) 
            DO UPDATE SET collected_at = CURRENT_TIMESTAMP
        """
        cur.execute(query, (sigungu_code, deal_ymd, data_type))
        conn.commit()
    finally:
        conn.close()


def fetch_real_price_from_api(sigungu_code, bjdong_code):
    """
    [수정됨] 수집 이력(Log)을 체크하여 불필요한 중복 호출 방지
    """
    print(f"   [System] 실거래가 데이터 최신화 점검 (시군구: {sigungu_code})...")

    # 법정동 코드 맵 로드 (API 호출 시 필요하므로 미리 로드)
    # 매번 로드하면 느리므로, 전역 변수나 캐싱이 되어 있다면 좋음
    code_map = get_bjdong_code_map()

    # 최근 3개월치 루프
    for i in range(10):
        target_date = datetime.now() - timedelta(days=30 * i)
        deal_ymd = target_date.strftime('%Y%m')

        # -------------------------------------------------------
        # 1. 매매(Trade) 데이터 수집
        # -------------------------------------------------------
        if not check_price_log(sigungu_code, deal_ymd, 'TRADE'):
            print(f"      [Fetch] 매매 데이터 수집: {deal_ymd}")
            try:
                fetch_trade_data_and_save(sigungu_code, deal_ymd, code_map)
                # [핵심] 결과가 있든 없든 로그를 남겨서 오늘 하루는 다시 조회 안 하게 함
                update_price_log(sigungu_code, deal_ymd, 'TRADE')
            except Exception as e:
                print(f"      [Error] 매매 데이터 수집 중 오류: {e}")
        else:
            # 이미 로그가 있으면 스킵
            pass

        # -------------------------------------------------------
        # 2. 전세(Rent) 데이터 수집
        # -------------------------------------------------------
        if not check_price_log(sigungu_code, deal_ymd, 'RENT'):
            print(f"      [Fetch] 전월세 데이터 수집: {deal_ymd}")
            try:
                fetch_rent_data_and_save(sigungu_code, deal_ymd, code_map)
                # [핵심] 수집 완료 로그 저장
                update_price_log(sigungu_code, deal_ymd, 'RENT')
            except Exception as e:
                print(f"      [Error] 전월세 데이터 수집 중 오류: {e}")
        else:
            pass

    print(f"   [System] 실거래가 점검 완료.")
    return True

def normalize_address(address):
    """
    주소 문자열의 앞부분(시/도)을 정식 명칭으로 변환하는 함수
    예: "인천 부평구..." -> "인천광역시 부평구..."
    """
    # 1. 줄임말 매핑 테이블 (필요한 만큼 추가 가능)
    sido_map = {
        "서울": "서울특별시","서울시": "서울특별시","인천": "인천광역시","인천시": "인천광역시","경기": "경기도",
        "부산": "부산광역시","대구": "대구광역시","광주": "광주광역시","대전": "대전광역시","울산": "울산광역시",
        "세종": "세종특별자치시","강원": "강원특별자치도","충북": "충청북도","충남": "충청남도","전북": "전북특별자치도",
        "전남": "전라남도","경북": "경상북도","경남": "경상남도","제주": "제주특별자치도"
    }

    # 2. 주소가 비어있으면 그대로 반환
    if not address or not isinstance(address, str):
        return address

    # 3. 공백 기준으로 단어 분리
    tokens = address.split()

    if not tokens:
        return address

    # 4. 첫 번째 단어(시/도)가 매핑 테이블에 있는지 확인하고 교체
    first_word = tokens[0]

    # "인천" -> "인천광역시"
    if first_word in sido_map:
        tokens[0] = sido_map[first_word]

    # 5. 다시 합쳐서 반환
    return " ".join(tokens)


def fetch_building_ledger_from_api(address, road_addr, target_pnu):
    """
    [수정됨] 전유부/표제부 수집 로직을 분리하여 호출하는 메인 함수
    """
    print(f"   [System] 수집 상태 점검: {address}")

    # =========================================================
    # 1. 정규화 및 상태 개별 확인
    # =========================================================
    address = normalize_address(address)
    road_addr = normalize_address(road_addr)

    is_exclusive_done = check_data_existence_by_pnu("building_info", target_pnu)
    is_title_done = check_data_existence_by_pnu("building_title_info", target_pnu)

    # 둘 다 완료된 경우 -> 즉시 종료
    if is_exclusive_done and is_title_done:
        print(f"      [Skip] 전유부/표제부 모두 이미 수집되어 있습니다.")
        return True, "이미 수집됨"

    # 하나라도 해야 할 일이 있으면 토큰 발급
    token = get_access_token()
    if not token: return False, "API 토큰 발급 실패"

    # =========================================================
    # 2. 전유부 수집 (필요시)
    # =========================================================
    if not is_exclusive_done:
        success = _collect_exclusive_with_retry(token, address)
        if not success:
            return False, "전유부 수집 실패 (데이터 없음)"
    else:
        print(f"      [Skip] 전유부 데이터 보유 중 (PNU 확인됨)")

    # =========================================================
    # 3. 표제부 수집 (필요시)
    # =========================================================
    if not is_title_done:
        success = _collect_title_with_retry(token, address)
        if not success:
            # 표제부 실패는 Critical하지 않다면 True로 넘길 수도 있지만,
            # 일단 요청하신 대로 False를 리턴하도록 구성했습니다.
            return False, "표제부 수집 실패 (데이터 없음)"
    else:
        print(f"      [Skip] 표제부 데이터 보유 중 (PNU 확인됨)")

    return True, "수집 및 저장 완료"

def get_real_time_data(address, deposit_amount):
    """
    [수정됨] 수집 상태(Log)를 먼저 확인하고, 필요 시 수집 후 데이터를 조회하는 로직
    """
    print(f"\n분석 요청: {address} (보증금: {deposit_amount:,}원)")

    # 1. 주소 변환 (지번 -> 도로명)
    # API 호출 및 DB 조회에 도로명 주소가 필수적으로 사용됨
    road_addr = get_road_address_from_kakao(address)

    # 주소 변환: 인천 => 인천광역시, 서울 => 서울특별시 등
    address = normalize_address(address)
    road_addr=normalize_address(road_addr)

    # 2. DB 조회 (이미 수집된 데이터인지 확인)
    # 여기서는 편의상 DB 쿼리로 가져오는 로직을 구현 (없으면 API 수집 로직 연결 필요)
    # 실제 서비스에선 API 수집 로직을 여기에 통합해야 함

    # 1. 주소 -> 고유번호(PNU) 변환 시도
    # (주의: bjd_map은 미리 로드되어 있어야 함)
    target_pnu, msg = convert_address_to_unique_no(address, bjd_map)

    # =========================================================
    # 2. [NEW] 실제 데이터 존재 여부 확인 (PNU 기반)
    # =========================================================
    is_exclusive_done = False
    is_title_done = False

    if target_pnu:
        # (1) 전유부(building_info) 확인
        is_exclusive_done = check_data_existence_by_pnu("building_info", target_pnu)

        # (2) 표제부(building_title_info) 확인
        is_title_done = check_data_existence_by_pnu("building_title_info", target_pnu)

        print(f"   [Status] 전유부: {'보유' if is_exclusive_done else '미보유'}, "
              f"표제부: {'보유' if is_title_done else '미보유'} (PNU: {target_pnu})")

    # 데이터가 부족하면 수집 수행 (토큰 발급은 내부에서 필요할 때만 함)
    if not (is_exclusive_done and is_title_done):
        print(f"   [Action] 데이터 미보유 -> API 수집 프로세스 진입")

        # 여기서 fetch_building_ledger_from_api를 호출하면
        # 내부에서 토큰 발급, 재시도, 부분 수집 등을 알아서 다 처리함
        success, msg = fetch_building_ledger_from_api(address, road_addr, target_pnu)

        if not success:
            return None, f"데이터 수집 실패: {msg}"
    else:
        print(f"   [Skip] 데이터 DB 보유 중 (API 호출 생략)")

    # =========================================================
    # 4. DB 데이터 조회 (쿼리)
    # =========================================================
    if target_pnu:
        where_clause = f"b.unique_number LIKE '{target_pnu}%'"
    else:
        where_clause = f"(b.road_address LIKE '%{road_addr}%' OR b.lot_address LIKE '%{address}%')"

    query = f"""
        SELECT 
            b.unique_number, b.detail_address, b.main_use, b.exclusive_area, 
            b.owner_name, b.ownership_changed_date, b.is_violating_building,
            p.price as PUBLIC_PRICE,
            t.household_cnt, t.parking_cnt, t.elevator_cnt, t.use_apr_day, t.is_violating as title_violation
        FROM building_info b
        LEFT JOIN public_price_history p ON b.id = p.building_info_id
        LEFT JOIN building_title_info t ON b.unique_number LIKE substr(t.unique_number, 1, 14) || '%' 
        WHERE {where_clause}
        ORDER BY p.base_date DESC LIMIT 1
    """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"   [Error] DB 조회 중 오류: {e}")
        df = pd.DataFrame()

    # 방금 수집했다고 해도 파싱 에러 등으로 데이터가 없을 수 있으므로 최종 확인
    if df.empty:
        return None, "수집은 완료되었으나 DB에서 데이터를 찾을 수 없습니다. (Parsing Error 가능성)"

    # 3. 변수 가공 (Data Processor 로직 재사용)
    row = df.iloc[0]

    # 건물 정보에서 법정동 코드 추출
    uniq_no = row['unique_number']  # 예: 2823710100-3-...
    sgg_code = ""
    bjd_code = ""
    bon = ""
    bu = ""

    if uniq_no:
        parts = uniq_no.split('-')
        if len(parts) >= 3:
            sgg_code = parts[0][:5]  # 28237 (시군구)
            bjd_code = parts[0][5:10]  # 10100 (법정동)
            bon = parts[2][:4]  # 본번
            bu = parts[2][4:8]  # 부번

    # 해당 번지의 최근 매매가 조회
    trade_query = f"""
        SELECT `거래금액(만원)` as price FROM raw_trade 
        WHERE 시군구='{sgg_code}' AND 법정동='{bjd_code}' 
          AND 본번='{bon}' AND 부번='{bu}'
        ORDER BY 계약일 DESC LIMIT 1
    """

    df_trade = pd.read_sql(trade_query, engine)

    # 매매가 없으면 API로 수집 시도
    if df_trade.empty and sgg_code:
        fetch_real_price_from_api(sgg_code, bjd_code)  # <-- 여기서 수집!
        df_trade = pd.read_sql(trade_query, engine)  # 다시 조회

    estimated_market_price = 0

    # 1. DB 실거래가 확인
    if not df_trade.empty:
        estimated_market_price = float(df_trade.iloc[0]['price'].replace(',', ''))
        print(f"   [Info] 실거래가 발견: {estimated_market_price}만원")
    else:
        # 2. 실거래가 없으면 주변 시세나 공시가 기반 추정 (이전 답변의 함수 활용 등)
        # 만약 공시가도 0이라면 일단 0으로 둠
        db_public_price_temp = float(row['PUBLIC_PRICE'] or 0) / 10000
        if db_public_price_temp > 0:
            estimated_market_price = db_public_price_temp * 1.4
        else:
            # 정말 아무것도 없으면 보증금 역산 (최후의 수단)
            estimated_market_price = (deposit_amount / 10000) / 0.8
            print(f"   [Warning] 데이터 전무 -> 보증금 기반 시세 추정")

    # 4. 시세 결정
    public_price = float(row['PUBLIC_PRICE'] or 0) / 10000

    if public_price == 0:
        # [수정된 로직] 보증금이 아니라 '찾아낸 실거래가'를 기준으로 역산
        if estimated_market_price > 0:
            public_price = estimated_market_price * 0.7  # 시세 반영률 70% 가정
            print(f"   [Fix] 공시가격 누락 -> 실거래가({estimated_market_price})의 70%로 추정: {public_price:.1f}만원")
        else:
            # 실거래가조차 못 구했을 때만 보증금 로직 사용
            public_price = (deposit_amount / 10000) * 0.7


    # (1) 기본 정보
    deposit = deposit_amount / 10000  # 만원 단위

    # (3) 비율 계산
    jeonse_ratio = deposit / estimated_market_price

    # HUG 한도 계산 (공시가가 0이었다면 추정된 public_price 사용)
    hug_limit = public_price * 1.26
    hug_risk_ratio = deposit / hug_limit

    # (4) 건물 정보
    # 나이
    use_apr_day = pd.to_datetime(row['use_apr_day'])
    building_age = (datetime.now() - use_apr_day).days / 365.25 if pd.notnull(use_apr_day) else 10

    # 주차
    house_cnt = row['household_cnt'] if row['household_cnt'] > 0 else 1
    parking_per_household = row['parking_cnt'] / house_cnt

    # 나홀로 아파트
    is_micro = 1 if house_cnt < 100 else 0

    # (5) 리스크 가중치 (간소화된 로직 적용)
    # 신탁
    is_trust = 1 if '신탁' in str(row['owner_name']) else 0

    # 단기 소유
    try:
        own_date = pd.to_datetime(row['ownership_changed_date'])
        own_days = (datetime.now() - own_date).days
    except:
        own_days = 9999

    short_term_w = 0.0
    if own_days < 90:
        short_term_w = 0.3
    elif own_days < 730:
        short_term_w = 0.15

    # 추정 대출 비율 (간략화)
    base_loan = 0.2
    type_w = 0.2 if '아파트' not in str(row['main_use']) else 0.0
    est_loan_ratio = min(0.9, base_loan + type_w + short_term_w + (0.3 if is_trust else 0))

    # 깡통전세 비율
    loan_amt = estimated_market_price * est_loan_ratio
    total_risk_ratio = (loan_amt + deposit) / estimated_market_price

    # (6) One-Hot Encoding 준비
    main_use = str(row['main_use'])
    use_dict = {col: 0 for col in USE_COLS}

    if '아파트' in main_use:
        use_dict['use_아파트'] = 1
    elif '오피스텔' in main_use:
        use_dict['use_오피스텔'] = 1
    elif '다세대' in main_use:
        use_dict['use_다세대주택'] = 1
    elif any(c in main_use for c in ['근린', '소매']):
        use_dict['use_근린생활시설'] = 1
    else:
        use_dict['use_기타'] = 1

    # 4. 최종 입력 데이터 생성
    input_data = {
        'jeonse_ratio': jeonse_ratio,
        'hug_risk_ratio': hug_risk_ratio,
        'total_risk_ratio': total_risk_ratio,
        'building_age': building_age,
        'parking_per_household': parking_per_household,
        'is_micro_complex': is_micro,
        'estimated_loan_ratio': est_loan_ratio,
        'is_trust_owner': is_trust,
        'short_term_weight': short_term_w
    }
    input_data.update(use_dict)  # 원핫 컬럼 병합

    return input_data, row['detail_address']


def predict_risk(address, deposit_amount):
    deposit_amount*=10000 # 만원 단위
    # 1. 데이터 준비
    input_dict, bldg_name = get_real_time_data(address, deposit_amount)

    if not input_dict:
        return {"error": bldg_name}  # 에러 메시지 반환

    # 2. DataFrame 변환 (모델 입력용)
    # 학습 때와 컬럼 순서를 완벽하게 맞춰야 함
    df_input = pd.DataFrame([input_dict])

    # [핵심 수정] 모델이 학습할 때 썼던 피처 순서를 그대로 가져와서 정렬합니다.
    # (모델 파일 안에 저장되어 있는 정답 순서입니다)
    try:
        # 학습된 피처 이름 목록 가져오기
        train_features = rf_model.feature_names_in_

        # 해당 순서대로 데이터프레임 재배열 (없는 컬럼은 0으로 채움)
        df_input = df_input.reindex(columns=train_features, fill_value=0)

    except AttributeError:
        # 구버전 scikit-learn이거나 피처 이름이 저장 안 된 경우 대비 (수동 지정)
        print("모델에서 피처 이름을 찾을 수 없어 수동 리스트를 사용합니다.")
        manual_features = [
            'jeonse_ratio', 'hug_risk_ratio', 'total_risk_ratio', 'building_age',
            'parking_per_household', 'is_micro_complex', 'estimated_loan_ratio',
            'is_trust_owner', 'short_term_weight',
            'use_아파트', 'use_오피스텔', 'use_다세대주택', 'use_근린생활시설', 'use_기타'
        ]
        df_input = df_input.reindex(columns=manual_features, fill_value=0)

    # 3. 예측 수행
    try:
        prob = rf_model.predict_proba(df_input)[0][1]  # 위험(1)일 확률
    except ValueError as e:
        return {"error": f"모델 예측 오류: 컬럼 불일치 ({e})"}
    is_risky = prob > 0.5

    # 4. 결과 반환
    return {
        "address": address,
        "building_name": bldg_name,
        "deposit": f"{int(deposit_amount / 10000)}만원",
        "risk_score": round(prob * 100, 2),
        "risk_level": "RISKY" if is_risky else "SAFE",
        "details": {
            "hug_ratio": round(input_dict['hug_risk_ratio'] * 100, 1),
            "total_ratio": round(input_dict['total_risk_ratio'] * 100, 1),
            "is_trust": bool(input_dict['is_trust_owner']),
            "is_short_term": bool(input_dict['short_term_weight'] > 0)
        }
    }


# --- 실행 테스트 ---
if __name__ == "__main__":
    # DB에 있는 실제 주소로 테스트해보세요
    test_address = "인천광역시 계양구 병방동 101-1"
    test_deposit = 4000  # 전세 보증금(만원 단위)

    result = predict_risk(test_address, test_deposit)

    import json

    print(json.dumps(result, indent=4, ensure_ascii=False))