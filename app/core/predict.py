import pandas as pd
import numpy as np
import joblib
import os
import sys
import re
from datetime import datetime, timedelta
from sqlalchemy import text
from app.core.config import engine

# --- 프로젝트 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- 모듈 임포트 (기존 유지) ---
from scripts.fetch_ledger_exclusive import (
    fetch_final_data_step, parse_and_save,
    get_dong_list_step, get_ho_list_step,
    get_access_token, fetch_initial_search,
    fetch_target_middle_unit, save_job_log
)
from scripts.fetch_ledger_title import fetch_step1_search, fetch_step2_detail, parse_and_save_title
from scripts.kakao_localmap_api import get_road_address_from_kakao, get_building_name_from_kakao, \
    get_all_address_and_building_from_kakao
from scripts.data_processor import _create_join_key_from_unique_no, _extract_floor_from_detail, engine
from scripts.db_manager import get_connection
from scripts.fetch_trade_data import fetch_trade_data_and_save, get_bjdong_code_map
from scripts.fetch_rent_data import fetch_rent_data_and_save

# --- 모델 로드 ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'fraud_rf_model.pkl')
try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"모델 로드 성공: {MODEL_PATH}")
except:
    print("모델 파일이 없습니다. 먼저 학습(train_model.py)을 실행하세요.")
    rf_model = None  # 모델 없이도 로직은 돌 수 있게 처리

# 법정동 코드 로드
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', '국토교통부_법정동코드_20250805.csv')
if os.path.exists(CSV_PATH):
    df_bjd = pd.read_csv(CSV_PATH, sep=',', encoding='cp949', dtype=str)
    bjd_map = dict(zip(df_bjd['법정동명'], df_bjd['법정동코드']))
else:
    bjd_map = {}
    print("Warning: 법정동 코드 파일이 없습니다.")

# [수정됨] 학습 모델의 Feature 컬럼명과 일치시켜야 함 (data_processor.py 참조)
# 기존 'use_아파트' -> 'type_APT' 등으로 변경
USE_COLS = ['type_APT', 'type_OFFICETEL', 'type_VILLA', 'type_ETC']

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


def save_prediction_result(meta, risk_level, risk_score, ai_prob):
    """
    예측 결과를 risk_analysis_result 테이블에 저장
    """
    try:
        # PNU를 이용해서 address_key 생성 (risk_pipeline과 포맷 통일)
        # 예: 2823710100-3-00650124 -> 28237-10100-0065-0124
        pnu = meta['unique_number']
        if pnu and '-' in pnu:
            parts = pnu.split('-')  # ['2823710100', '3', '00650124']
            sgg_bjd = parts[0]
            bon_bu = parts[2]
            address_key = f"{sgg_bjd[:5]}-{sgg_bjd[5:]}-{bon_bu[:4]}-{bon_bu[4:]}"
        else:
            address_key = "UNKNOWN"

        # SQL 파라미터 준비
        params = {
            "building_info_id": meta['building_info_id'],
            "address_key": address_key,
            "used_rent_price": meta['rent_price'] / 10000,  # 만원 단위 저장
            "used_market_price": meta['est_market_price'] / 10000,  # 만원 단위 저장
            "jeonse_ratio": meta['jeonse_ratio'],
            "hug_safe_limit": meta['hug_safe_limit'] / 10000,
            "hug_risk_ratio": meta['hug_risk_ratio'],
            "total_risk_ratio": meta['total_risk_ratio'],
            "estimated_loan_amount": 0,  # 요청하신 대로 0으로 저장
            "risk_level": risk_level,
            "risk_score": risk_score,  # 0~100 점수
            "ai_risk_prob": ai_prob  # 0.0~1.0 확률
        }

        # 3. [핵심] 삭제 후 삽입 (Transaction)
        # 같은 address_key를 가진 데이터가 있다면 먼저 삭제합니다.
        sql_delete = text("DELETE FROM risk_analysis_result WHERE address_key = :address_key")

        # INSERT 쿼리 (중복 시 업데이트 or 무시 전략 선택)
        # 여기서는 created_at을 갱신하며 재저장하는 방식으로 작성
        sql_insert = text("""
            INSERT INTO risk_analysis_result (
                building_info_id, address_key, used_rent_price, used_market_price,
                jeonse_ratio, hug_safe_limit, hug_risk_ratio, total_risk_ratio,
                estimated_loan_amount, risk_level, risk_score, ai_risk_prob, created_at
            ) VALUES (
                :building_info_id, :address_key, :used_rent_price, :used_market_price,
                :jeonse_ratio, :hug_safe_limit, :hug_risk_ratio, :total_risk_ratio,
                :estimated_loan_amount, :risk_level, :risk_score, :ai_risk_prob, CURRENT_TIMESTAMP
            )
        """)

        with engine.connect() as conn:
            with conn.begin():  # 트랜잭션 시작 (둘 다 성공하거나 둘 다 실패하거나)
                # 1) 기존 이력 삭제
                conn.execute(sql_delete, {"address_key": address_key})

                # 2) 최신 이력 저장
                conn.execute(sql_insert, params)

            print(f"   [DB] 기존 이력 갱신 및 저장 완료 ({address_key})")

    except Exception as e:
        print(f"   [Error] 결과 저장 실패: {e}")

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

# =========================================================
# 헬퍼 함수들 (기존 로직 유지)
# =========================================================
def normalize_address(address):
    sido_map = {
        "서울": "서울특별시", "서울시": "서울특별시", "인천": "인천광역시", "인천시": "인천광역시", "경기": "경기도",
        "부산": "부산광역시", "대구": "대구광역시", "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시",
        "세종": "세종특별자치시", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도",
        "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"
    }
    if not address or not isinstance(address, str): return address
    tokens = address.split()
    if not tokens: return address
    if tokens[0] in sido_map: tokens[0] = sido_map[tokens[0]]
    return " ".join(tokens)

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

def convert_address_to_unique_no(address, bjd_mapping):
    match = re.search(r'(.+)\s+(\d+(?:-\d+)?)', address)
    if not match: return None, "주소 형식 오류"
    region_addr = match.group(1).strip()
    bunji_str = match.group(2).strip()
    if region_addr not in bjd_mapping: return None, "법정동 코드 없음"
    bjd_code = bjd_mapping[region_addr]
    if '-' in bunji_str:
        main_no, sub_no = bunji_str.split('-')
    else:
        main_no, sub_no = bunji_str, "0"
    return f"{bjd_code}-3-{int(main_no):04d}{int(sub_no):04d}", "성공"


def check_data_existence_by_pnu(table_name, unique_number_prefix):
    if not unique_number_prefix: return False
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT 1 FROM {table_name} WHERE unique_number LIKE ? LIMIT 1", (unique_number_prefix + '%',))
        return cur.fetchone() is not None
    finally:
        conn.close()


# ... (fetch 관련 함수들은 기존 로직과 동일하므로 생략하거나, 기존 파일의 import 사용) ...
# 여기서는 핵심인 get_real_time_data 위주로 수정합니다.

def fetch_real_price_if_needed(sgg_code, bjd_code):
    """실거래가 수집이 필요한 경우 실행"""
    # (간소화된 로직: 실제 구현은 기존 코드의 check_price_log 활용)
    try:
        deal_ymd = datetime.now().strftime('%Y%m')
        # fetch_trade_data_and_save(sgg_code, deal_ymd, get_bjdong_code_map()) # 실제 호출 시 주석 해제
        return True
    except:
        return False


# =========================================================
# [핵심] 실시간 데이터 조회 및 가공
# =========================================================
def get_real_time_data(address, deposit_amount):
    print(f"\n분석 요청: {address} (보증금: {deposit_amount:,}원)")

    # 1. 주소 정규화
    address, road_addr, building_name = get_all_address_and_building_from_kakao(address)
    address = normalize_address(address)
    road_addr = normalize_address(road_addr)

    # 2. PNU 생성 및 DB 조회
    target_pnu, msg = convert_address_to_unique_no(address, bjd_map)

    # DB 존재 여부 확인 변수 초기화
    is_exclusive_done = False
    is_title_done = False

    # (1) PNU가 생성되었다면 DB를 먼저 조회해봅니다.
    if target_pnu:
        is_exclusive_done = check_data_existence_by_pnu("building_info", target_pnu)
        is_title_done = check_data_existence_by_pnu("building_title_info", target_pnu)

        print(f"   [Status] 전유부: {'보유' if is_exclusive_done else '미보유'}, "
              f"표제부: {'보유' if is_title_done else '미보유'} (PNU: {target_pnu})")

    # (2) 전유부나 표제부 중 하나라도 없으면 API 수집을 시도합니다.
    if not (is_exclusive_done and is_title_done):
        print(f"   [Action] 데이터 미보유 -> API 수집 프로세스 진입")

        # 앞서 정의해둔 수집 함수 호출 (내부에서 토큰 발급, 재시도 등 처리)
        success, fetch_msg = fetch_building_ledger_from_api(address, road_addr, target_pnu)

        if not success:
            print(f"   [Fail] 수집 실패: {fetch_msg}")
            # 수집에 실패하면 분석을 진행할 수 없으므로 종료
            return None, f"데이터 수집 실패: {fetch_msg}"

        print(f"   [Success] 수집 및 DB 저장 완료")
    else:
        print(f"   [Skip] 데이터 DB 보유 중 (API 호출 생략)")

    # 3. DB 조회
    if target_pnu:
        where_clause = f"b.unique_number LIKE '{target_pnu}%'"
    else:
        where_clause = f"(b.road_address LIKE '%{road_addr}%' OR b.lot_address LIKE '%{address}%')"

    query = f"""
        SELECT 
            b.id as building_info_id,
            b.unique_number, b.detail_address, b.main_use, b.exclusive_area, 
            b.owner_name, b.ownership_changed_date, b.is_violating_building,
            p.price as PUBLIC_PRICE,
            t.household_cnt, t.parking_cnt, t.elevator_cnt, t.use_apr_day
        FROM building_info b
        LEFT JOIN public_price_history p ON b.id = p.building_info_id
        LEFT JOIN building_title_info t ON b.unique_number LIKE substr(t.unique_number, 1, 14) || '%' 
        WHERE {where_clause}
        ORDER BY p.base_date DESC LIMIT 1
    """

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"DB Error: {e}")
        return None, "DB 조회 오류"

    if df.empty:
        return None, "건물 정보를 찾을 수 없습니다. (데이터 미수집)"

    row = df.iloc[0]

    # 4. 실거래가 조회 (df_trade 오류 수정 부분)
    # [수정 1] df_trade를 미리 빈 DataFrame으로 초기화하여 참조 오류 방지
    df_trade = pd.DataFrame()

    sgg_code = ""
    bjd_code = ""
    bon = ""
    bu = ""

    if row['unique_number']:
        parts = row['unique_number'].split('-')
        if len(parts) >= 3:
            sgg_code = parts[0][:5]
            bjd_code = parts[0][5:10]
            bon = parts[2][:4]
            bu = parts[2][4:8]

    if sgg_code:
        trade_query = f"""
            SELECT 거래금액 as price FROM raw_trade 
            WHERE 시군구='{sgg_code}' AND 법정동='{bjd_code}' 
              AND 본번='{bon}' AND 부번='{bu}'
            ORDER BY 계약일 DESC LIMIT 1
        """
        df_trade = pd.read_sql(trade_query, engine)

        # 매매가 없으면 API로 수집 시도
        if df_trade.empty and sgg_code:
            fetch_real_price_from_api(sgg_code, bjd_code)  # <-- 여기서 수집!
            df_trade = pd.read_sql(trade_query, engine)  # 다시 조회

    # 5. 시세 추정
    estimated_market_price = 0

    # (1) 실거래가 우선
    if not df_trade.empty:
        try:
            val = str(df_trade.iloc[0]['price']).replace(',', '')
            estimated_market_price = float(val)
            print(f"   [Info] 실거래가 발견: {estimated_market_price}만원")
        except:
            pass

    # (2) 공시지가 기반 추정 (실거래가 없을 시)
    public_price = float(row['PUBLIC_PRICE'] or 0) / 10000

    if estimated_market_price == 0:
        if public_price > 0:
            # 빌라 등은 1.5배~1.8배 적용 (data_processor와 로직 통일)
            m_use = str(row['main_use'])
            if any(x in m_use for x in ['다세대', '오피스텔', '연립', '근린']):
                estimated_market_price = public_price * 1.8
            else:
                estimated_market_price = public_price * 1.5
            print(f"   [Info] 공시가 기반 시세 추정: {estimated_market_price:.1f}만원")
        else:
            # (3) 최후의 수단: 보증금 역산
            estimated_market_price = (deposit_amount / 10000) / 0.8
            print(f"   [Warning] 데이터 부족 -> 보증금 기반 시세 역산")

    # HUG 한도용 공시가 보정
    if public_price == 0:
        public_price = estimated_market_price * 0.7

    # 6. 피처 엔지니어링 (data_processor.py와 동일하게 구성)
    deposit = deposit_amount / 10000  # 만원 단위

    # (1) 재무적 위험 (순수 전세가율)
    jeonse_ratio = deposit / estimated_market_price

    # (2) HUG 위험도
    hug_limit = public_price * 1.26
    hug_risk_ratio = deposit / hug_limit if hug_limit > 0 else 0

    # (3) 정성적 위험 점수 (estimated_loan_ratio -> risk_factor_score)
    # 신탁
    is_trust = 1 if '신탁' in str(row['owner_name']) else 0

    # 단기 소유
    try:
        own_date = pd.to_datetime(row['ownership_changed_date'])
        days_diff = (datetime.now() - own_date).days
    except:
        days_diff = 9999

    short_term_w = 0.0
    if days_diff < 90:
        short_term_w = 0.3
    elif days_diff < 730:
        short_term_w = 0.1

    # 건물 유형 가중치
    main_use = str(row['main_use'])
    type_w = 0.0
    if '근린' in main_use:
        type_w = 0.4
    elif any(c in main_use for c in ['다세대', '오피스텔', '연립']):
        type_w = 0.1

    # 위험 점수 합산 (가짜 빚이 아님!)
    val = type_w + short_term_w + (0.5 if is_trust else 0)
    risk_score_val = np.clip(val, 0, 1.0)

    # (4) 종합 리스크 비율 (순수 전세가율 + 실제 등기부 대출)
    real_debt = 0  # 현재 등기부 채권 데이터가 없으므로 0
    total_risk_ratio = (deposit + real_debt) / estimated_market_price

    # (5) 나머지 피처
    house_cnt = row['household_cnt'] if row['household_cnt'] and row['household_cnt'] > 0 else 1
    parking = (row['parking_cnt'] or 0) / house_cnt
    is_micro = 1 if house_cnt < 100 else 0

    use_apr = pd.to_datetime(row['use_apr_day'])
    age = (datetime.now() - use_apr).days / 365.25 if pd.notnull(use_apr) else 10
    is_illegal = 1 if str(row['is_violating_building']).strip() == 'Y' else 0

    # (6) One-Hot Encoding (수동 매핑)
    # [수정 2] data_processor.py의 simplify_use 로직과 컬럼명 일치시킴
    type_dict = {col: 0 for col in USE_COLS}

    if '아파트' in main_use:
        type_dict['type_APT'] = 1
    elif '오피스텔' in main_use:
        type_dict['type_OFFICETEL'] = 1
    elif '다세대' in main_use or '연립' in main_use:
        type_dict['type_VILLA'] = 1
    else:
        type_dict['type_ETC'] = 1  # 근린 등은 기타로 분류하거나 로직에 따라 조정

    # 최종 입력 데이터
    input_data = {
        'jeonse_ratio': jeonse_ratio,
        'hug_risk_ratio': hug_risk_ratio,
        'total_risk_ratio': total_risk_ratio,
        'estimated_loan_ratio': risk_score_val,  # 컬럼명은 유지하되 내용은 '점수'
        'building_age': age,
        'is_illegal': is_illegal,
        'parking_per_household': parking,
        'is_micro_complex': is_micro,
        'is_trust_owner': is_trust,      # 학습 모델 피처 목록에 따라 추가/제거 결정
        'short_term_weight': short_term_w
    }
    input_data.update(type_dict)

    meta_data = {
        "building_info_id": int(row['building_info_id']),
        "unique_number": row['unique_number'],
        "rent_price": deposit_amount,  # 원단위
        "est_market_price": estimated_market_price * 10000,  # 원단위 환산
        "jeonse_ratio": jeonse_ratio,
        "hug_safe_limit": hug_limit * 10000,  # 원단위 환산
        "hug_risk_ratio": hug_risk_ratio,
        "total_risk_ratio": total_risk_ratio,
        "risk_score_val": risk_score_val  # estimated_loan_ratio 컬럼에 들어갈 점수
    }

    return input_data, row['detail_address'], meta_data


def predict_risk(address, deposit_amount):
    deposit_amount *= 10000  # 만원 단위 환산

    # 1. 데이터 준비
    # input_dict: AI 입력용, bldg_name: 표시용, meta: DB 저장용
    data = get_real_time_data(address, deposit_amount)

    if not data or data[0] is None:
        return {"error": data[1] if data else "데이터 조회 실패"}

    input_dict, bldg_name, meta = data

    # 2. DataFrame 변환
    df_input = pd.DataFrame([input_dict])

    # 3. 컬럼 순서 맞추기 (모델이 학습된 순서대로)
    if rf_model:
        try:
            train_features = rf_model.feature_names_in_
            # 없는 컬럼은 0으로 채우고, 순서 재배열
            df_input = df_input.reindex(columns=train_features, fill_value=0)
        except:
            print("모델 피처 이름 확인 불가. 수동 리스트 사용 권장.")
            pass

        # 4. 예측 수행
        try:
            prob = rf_model.predict_proba(df_input)[0][1]
        except Exception as e:
            return {"error": f"모델 예측 오류: {e}"}
    else:
        # 모델 파일이 없을 경우 룰베이스로 대체
        prob = 0.0
        if input_dict['total_risk_ratio'] >= 0.8:
            prob = 0.9
        elif input_dict['estimated_loan_ratio'] >= 0.5:
            prob = 0.7

    # 5. 등급 판정
    if prob < 0.4:
        risk_level = "SAFE"
    elif prob < 0.7:
        risk_level = "CAUTION"
    else:
        risk_level = "RISKY"

    risk_score_display = round(float(prob * 100), 2)

    save_prediction_result(
        meta=meta,
        risk_level=risk_level,
        risk_score=risk_score_display,
        ai_prob=float(prob)
    )

    return {
        "address": address,
        "building_name": bldg_name,
        "deposit": f"{int(deposit_amount / 10000)}만원",

        "risk_score": round(float(prob * 100), 2),
        "risk_level": risk_level,
        "details": {
            "hug_ratio": round(float(input_dict['hug_risk_ratio'] * 100), 1),
            "total_ratio": round(float(input_dict['total_risk_ratio'] * 100), 1),
            "is_trust": bool(input_dict.get('is_trust_owner', 0)),
            "is_short_term": bool(input_dict.get('short_term_weight', 0) > 0)
        }
    }

if __name__ == "__main__":
    # 테스트
    test_addr = "인천광역시 부평구 산곡동 145"
    print(predict_risk(test_addr, 20000))