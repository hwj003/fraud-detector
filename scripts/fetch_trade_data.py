# fraud_detector_project/scripts/fetch_trade_data.py

import requests, pandas as pd, xml.etree.ElementTree as ET
import os, sys, time
from datetime import datetime
from sqlalchemy import text
from functools import lru_cache

# --- 프로젝트 경로 및 엔진 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from app.core.config import engine, load_dotenv

# --- 환경 변수 및 상수 ---
load_dotenv()
API_SERVICE_KEY = os.getenv("API_SERVICE_KEY")
API_URLS_TRADE = {
    "APT": "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade",
    "RH": "https://apis.data.go.kr/1613000/RTMSDataSvcRHTrade/getRTMSDataSvcRHTrade", # 연립다세대
    "OFFI": "https://apis.data.go.kr/1613000/RTMSDataSvcOffiTrade/getRTMSDataSvcOffiTrade" # 오피스텔
}
TRADE_TABLE_NAME = "raw_trade"
REGION_TABLE_NAME = "meta_sgg_codes"
LEGAL_CODES_CSV_PATH = os.path.join(project_root, 'data', '국토교통부_법정동코드_20250805.csv')  # [신규]

OLDEST_DATE_YMD = "202301"
API_CALL_LIMIT_PER_RUN = 9900
SLEEP_TIME_BETWEEN_CALLS = 0.5


# --- [신규] 헬퍼: 법정동 코드(CSV) 로드 및 매핑 딕셔너리 생성 ---
@lru_cache(maxsize=1)
def get_bjdong_code_map() -> dict:
    """
    'legal_codes.csv'를 로드하여 (시군구코드, 읍면동이름) -> (법정동코드) 매핑 딕셔너리 생성
    예: { ('11110', '청운동'): '10100', ('11110', '신교동'): '10200' }
    """
    print("--- [헬퍼] 법정동 코드 마스터(CSV) 로드 중... (1회 실행) ---")
    try:
        df = pd.read_csv(LEGAL_CODES_CSV_PATH, sep=',', encoding='cp949', dtype=str)
    except Exception as e:
        print(f"[치명적 오류] 'legal_codes.csv' 로드 실패: {e}")
        return {}

    df = df.rename(columns={'법정동코드': 'code', '법정동명': 'name', '폐지여부': 'status'})
    df_active = df[df['status'] == '존재'].copy()

    # 10자리 전체 코드에서 '읍면동' 레벨(00000으로 끝나지 않음)만 필터링
    df_dong_level = df_active[~df_active['code'].str.endswith('00000')].copy()

    df_dong_level['sgg_code'] = df_dong_level['code'].str.slice(0, 5)
    df_dong_level['bjdong_code'] = df_dong_level['code'].str.slice(5, 10)

    # [주의] '법정동명'에서 시/도/구 이름 부분 제거 (e.g., "서울특별시 종로구 청운동" -> "청운동")
    # (API가 'umdNm'으로 '청운동'만 반환한다고 가정)
    df_dong_level['dong_name_only'] = df_dong_level['name'].str.split().str[-1]

    # (시군구코드, 읍면동이름)을 Key로, (법정동코드)를 Value로 하는 딕셔너리 생성
    code_map = {}
    for row in df_dong_level.itertuples():
        key = (row.sgg_code, row.dong_name_only)
        code_map[key] = row.bjdong_code

    print(f"--- [헬퍼] 법정동 코드 맵({len(code_map)}개) 생성 완료 ---")
    return code_map


# --- 1. XML 파싱 함수 ---
def parse_trade_xml_to_df(xml_text: str, code_map: dict) -> pd.DataFrame:
    """매매 실거래가 API의 XML 응답을 DataFrame으로 파싱합니다."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame()
    if root.findtext('.//resultCode', '99') not in ('00', '000'): return pd.DataFrame()
    items = root.findall('.//item')
    if not items: return pd.DataFrame()

    records = []
    for item in items:
        bonbeon, bubeon = '0000', '0000'
        jibun_str = item.findtext('jibun', '').strip()

        if not jibun_str:
            continue

        if jibun_str:
            parts = jibun_str.split('-')
            bonbeon = parts[0].lstrip('0').strip().zfill(4) or '0000'
            if len(parts) > 1:
                bubeon = parts[1].lstrip('0').strip().zfill(4) or '0000'
        deal_date = f"{item.findtext('dealYear', '')}{item.findtext('dealMonth', '').zfill(2)}{item.findtext('dealDay', '').zfill(2)}"
        deal_amount_str = item.findtext('dealAmount', '0').replace(',', '').strip()

        # 이름(umdNm) -> 코드(bjdongCd) 변환
        sgg_code = item.findtext('sggCd').strip()
        dong_name = item.findtext('umdNm', '').strip()

        # 맵에서 (시군구코드, 읍면동이름) 키로 법정동코드를 찾음
        bjdong_code = code_map.get((sgg_code, dong_name))

        if bjdong_code is None:
            # print(f"  - [경고] 법정동 코드를 찾을 수 없습니다: ({sgg_code}, {dong_name})")
            continue  # 매핑 실패한 데이터는 저장하지 않음

        record = {
            '시군구': sgg_code,
            '법정동': bjdong_code,
            '본번': bonbeon,
            '부번': bubeon,
            '거래금액(만원)': deal_amount_str,
            '계약일': deal_date
        }
        records.append(record)
    return pd.DataFrame(records)


# --- 2. API 호출 함수 ---
def fetch_trade_data_and_save(lawd_cd: str, deal_ymd: str, code_map: dict) -> bool:
    """ 모든 유형(아파트, 빌라, 오피스텔)의 API를 호출하고 통합 저장"""

    all_dfs = [] # 모든 API 결과를 취합할 리스트

    # 3개 API URL을 순회
    for building_type, api_url in API_URLS_TRADE.items():
        params = {'serviceKey': API_SERVICE_KEY, 'LAWD_CD': lawd_cd, 'DEAL_YMD': deal_ymd, 'numOfRows': '1000'}
        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()

            # 공통 파싱 함수 사용
            df_api_data = parse_trade_xml_to_df(response.text, code_map)

            if not df_api_data.empty:
                print(f"  -> {building_type}: {len(df_api_data)} 건 수집됨.")
                all_dfs.append(df_api_data)
            else:
                print(f"  -> {building_type}: 데이터 없음 (0건).")

        except Exception as e:
            print(f"  -> {building_type} API 처리 실패: {e}")
            continue

    # 모든 결과를 하나로 합쳐서 DB에 저장
    if not all_dfs:
        print("  -> 최종 저장: 0건.")
        return True

    try:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        # 중복 데이터 제거
        df_combined = df_combined.drop_duplicates()

        df_combined.to_sql(TRADE_TABLE_NAME, con=engine, if_exists='append', index=False)
        print(f"  -> 최종 저장: {len(df_combined)} 건 완료.")
        return True
    except Exception as e:
        print(f"  -> 최종 DB 저장 실패: {e}")
        return False


# --- 3. DB 관리 함수 ---
def get_regions_to_fetch_from_db() -> list:
    try:
        with engine.connect() as conn:
            query = f"""
                SELECT sgg_code, trade_last_fetched_date 
                FROM {REGION_TABLE_NAME}
                WHERE trade_last_fetched_date >= '{OLDEST_DATE_YMD}'
                ORDER BY sgg_code ASC
            """
            df_regions = pd.read_sql(query, con=conn)
            return list(df_regions.itertuples(index=False, name=None))
    except Exception as e:
        print(f"[오류] DB에서 지역 목록(매매)을 불러오는 데 실패했습니다: {e}")
        return []


def update_fetch_progress_in_db(region_code: str, date_ym: str):
    # (이하 'trade_last_fetched_date' 업데이트 로직 동일)
    try:
        with engine.connect() as conn:
            query = text(f"""
                UPDATE {REGION_TABLE_NAME}
                SET trade_last_fetched_date = :date_ym
                WHERE sgg_code = :region_code
            """)
            conn.execute(query, {"date_ym": date_ym, "region_code": region_code})
            conn.commit()
    except Exception as e:
        print(f"[경고] DB 진행 상황(매매) 업데이트 실패 (region: {region_code}): {e}")


# --- 4. 메인 실행 루프 ---
def main_fetch_loop():
    """'라운드 로빈' 방식으로 매매 데이터를 수집합니다."""
    call_count = 0
    oldest_date_dt = pd.to_datetime(OLDEST_DATE_YMD, format='%Y%m')

    # 스크립트 시작 시 1회만 법정동 코드 맵을 로드
    code_map = get_bjdong_code_map()
    if not code_map:
        print("[치명적 오류] 법정동 코드 맵을 생성할 수 없어 스크립트를 종료합니다.")
        return

    print(f"--- [매매] DB 기반 [라운드 로빈] 데이터 수집을 시작합니다. ---")
    try:
        while call_count < API_CALL_LIMIT_PER_RUN:
            print(f"\n--- [매매] 새 수집 라운드 시작 (현재 호출: {call_count}) ---")
            regions = get_regions_to_fetch_from_db()
            if not regions: break

            work_done_in_this_round = False
            for region_code, last_fetched_date_str in regions:
                if call_count >= API_CALL_LIMIT_PER_RUN: break

                date_to_fetch_dt = (pd.to_datetime(last_fetched_date_str, format='%Y%m') -
                                    pd.DateOffset(months=1))
                if date_to_fetch_dt < oldest_date_dt: continue

                work_done_in_this_round = True
                date_ym_str = date_to_fetch_dt.strftime('%Y%m')
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] 수집 시도: {region_code}-{date_ym_str} (호출 {call_count + 1})")
                call_count += 1

                # [수정] code_map 인자 전달
                success = fetch_trade_data_and_save(region_code, date_ym_str, code_map)

                if success:
                    update_fetch_progress_in_db(region_code, date_ym_str)
                else:
                    print(f"  -> [경고] {region_code}-{date_ym_str} 처리 실패.")
                time.sleep(SLEEP_TIME_BETWEEN_CALLS)

            if call_count >= API_CALL_LIMIT_PER_RUN: break
            if not work_done_in_this_round:
                print("이번 라운드에서 수집할 데이터가 없었습니다.")
                break
    except KeyboardInterrupt:
        print("\n[중단] 사용자에 의해 스크립트가 중지되었습니다.")
    print("\n--- [매매] 데이터 수집 세션이 완료되었습니다. ---")


if __name__ == "__main__":
    main_fetch_loop()