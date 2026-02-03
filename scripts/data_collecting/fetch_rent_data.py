"""
전월세 실거래가 데이터 수집 스크립트

국토교통부 API를 통해 아파트, 연립다세대, 오피스텔의
전월세 실거래가 데이터를 수집하여 DB에 저장합니다.

사용법:
    # 프로젝트 루트에서 실행
    python -m scripts.fetch_data.fetch_rent_data

    # 또는 직접 실행
    python scripts/fetch_data/fetch_rent_data.py
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from sqlalchemy import text
from dotenv import load_dotenv

# =============================================================================
# 경로 설정 (프로젝트 루트 기준)
# =============================================================================
# 이 파일의 위치: fraud-detector/scripts/fetch_data/fetch_rent_data.py
# 프로젝트 루트: fraud-detector/
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/fetch_data/
SCRIPTS_DIR = SCRIPT_DIR.parent               # scripts/
PROJECT_ROOT = SCRIPTS_DIR.parent             # fraud-detector/

# 프로젝트 루트를 Python 경로에 추가 (app 모듈 import를 위해)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# app 모듈 import (경로 설정 후)
# =============================================================================
from app.core import get_engine

# =============================================================================
# 환경 변수 로드
# =============================================================================
load_dotenv(PROJECT_ROOT / '.env')
API_SERVICE_KEY = os.getenv("API_SERVICE_KEY")

# =============================================================================
# 상수 정의
# =============================================================================
API_URLS_RENT = {
    "아파트": "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent",
    "연립다세대": "https://apis.data.go.kr/1613000/RTMSDataSvcRHRent/getRTMSDataSvcRHRent",
    "오피스텔": "https://apis.data.go.kr/1613000/RTMSDataSvcOffiRent/getRTMSDataSvcOffiRent"
}

RENT_TABLE_NAME = "raw_rent"
REGION_TABLE_NAME = "meta_sgg_codes"

# 데이터 파일 경로 (프로젝트 루트 기준)
LEGAL_CODES_CSV_PATH = PROJECT_ROOT / 'data' / '국토교통부_법정동코드_20250805.csv'

OLDEST_DATE_YMD = "202301"
API_CALL_LIMIT_PER_RUN = 9900
SLEEP_TIME_BETWEEN_CALLS = 0.5


# =============================================================================
# 헬퍼 함수: 법정동 코드 로드
# =============================================================================
@lru_cache(maxsize=1)
def get_bjdong_code_map() -> dict:
    """
    법정동 코드 CSV를 로드하여 (시군구코드, 동이름) -> 법정동코드 매핑 생성

    Returns:
        dict: {(sgg_code, dong_name): bjdong_code} 형태의 딕셔너리
    """
    print("--- [헬퍼] 법정동 코드 마스터(CSV) 로드 중... (1회 실행) ---")

    if not LEGAL_CODES_CSV_PATH.exists():
        print(f"[치명적 오류] 법정동 코드 파일이 없습니다: {LEGAL_CODES_CSV_PATH}")
        return {}

    try:
        df = pd.read_csv(LEGAL_CODES_CSV_PATH, sep=',', encoding='cp949', dtype=str)
    except Exception as e:
        print(f"[치명적 오류] 법정동 코드 CSV 로드 실패: {e}")
        return {}

    df = df.rename(columns={
        '법정동코드': 'code',
        '법정동명': 'name',
        '폐지여부': 'status'
    })

    # 존재하는 동 단위만 필터링
    df_active = df[df['status'] == '존재'].copy()
    df_dong_level = df_active[~df_active['code'].str.endswith('00000')].copy()

    df_dong_level['sgg_code'] = df_dong_level['code'].str.slice(0, 5)
    df_dong_level['bjdong_code'] = df_dong_level['code'].str.slice(5, 10)
    df_dong_level['dong_name_only'] = df_dong_level['name'].str.split().str[-1]

    code_map = {}
    for row in df_dong_level.itertuples():
        key = (row.sgg_code, row.dong_name_only)
        code_map[key] = row.bjdong_code

    print(f"--- [헬퍼] 법정동 코드 맵({len(code_map)}개) 생성 완료 ---")
    return code_map


# =============================================================================
# XML 파싱 함수
# =============================================================================
def parse_rent_xml_to_df(xml_text: str, code_map: dict, building_type: str) -> pd.DataFrame:
    """
    국토부 API 응답 XML을 DataFrame으로 변환

    Args:
        xml_text: API 응답 XML 문자열
        code_map: 법정동 코드 매핑 딕셔너리
        building_type: 건물 유형 (아파트, 연립다세대, 오피스텔)

    Returns:
        pd.DataFrame: 파싱된 전월세 데이터
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame()

    if root.findtext('.//resultCode', '99') not in ('00', '000'):
        return pd.DataFrame()

    items = root.findall('.//item')
    if not items:
        return pd.DataFrame()

    # 건물 유형별 건물명 태그 매핑
    name_tag_map = {
        '아파트': 'aptNm',
        '오피스텔': 'offiNm',
        '연립다세대': 'mhouseNm'
    }
    name_tag = name_tag_map.get(building_type, 'aptNm')

    records = []
    for item in items:
        # 지번 파싱
        jibun_str = item.findtext('jibun', '').strip()
        if not jibun_str:
            continue

        bonbeon, bubeon = '0000', '0000'
        parts = jibun_str.split('-')
        bonbeon = parts[0].lstrip('0').strip().zfill(4) or '0000'
        if len(parts) > 1:
            bubeon = parts[1].lstrip('0').strip().zfill(4) or '0000'

        # 계약일
        deal_date = (
            f"{item.findtext('dealYear', '')}"
            f"{item.findtext('dealMonth', '').zfill(2)}"
            f"{item.findtext('dealDay', '').zfill(2)}"
        )

        # 금액 파싱
        deposit_str = item.findtext('deposit', '0').replace(',', '').strip()
        rent_str = item.findtext('monthlyRent', '0').replace(',', '').strip()

        # 기타 정보
        floor_str = item.findtext('floor', '1').strip()
        sgg_code = item.findtext('sggCd', '').strip()
        dong_name = item.findtext('umdNm', '').strip()

        # 법정동 코드 매핑
        bjdong_code = code_map.get((sgg_code, dong_name))
        if bjdong_code is None:
            continue

        record = {
            '시군구': sgg_code,
            '법정동': bjdong_code,
            '본번': bonbeon,
            '부번': bubeon,
            '보증금': deposit_str,
            '월세': rent_str,
            '계약일': deal_date,
            '계약유형': item.findtext('contractType', '').strip(),
            '건물유형': building_type,
            '층': floor_str,
            '전용면적': item.findtext('excluUseAr', ''),
            '건물명': item.findtext(name_tag, ''),
            '건축년도': item.findtext('buildYear', '')
        }
        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# API 호출 및 저장
# =============================================================================
def fetch_rent_data_and_save(lawd_cd: str, deal_ymd: str, code_map: dict) -> bool:
    """
    모든 건물 유형의 전월세 데이터를 수집하여 DB에 저장

    Args:
        lawd_cd: 시군구 코드
        deal_ymd: 거래년월 (YYYYMM)
        code_map: 법정동 코드 매핑

    Returns:
        bool: 성공 여부
    """
    engine = get_engine()
    all_dfs = []

    for building_type, api_url in API_URLS_RENT.items():
        params = {
            'serviceKey': API_SERVICE_KEY,
            'LAWD_CD': lawd_cd,
            'DEAL_YMD': deal_ymd,
            'numOfRows': '1000'
        }

        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()

            df_api_data = parse_rent_xml_to_df(response.text, code_map, building_type)

            if not df_api_data.empty:
                print(f"  -> {building_type}: {len(df_api_data)} 건 수집됨.")
                all_dfs.append(df_api_data)

        except requests.RequestException as e:
            print(f"  -> {building_type} API 요청 실패: {e}")
            continue
        except Exception as e:
            print(f"  -> {building_type} 처리 실패: {e}")
            continue

    if not all_dfs:
        print("  -> 최종 저장: 0건.")
        return True

    try:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_combined = df_combined.drop_duplicates()

        df_combined.to_sql(RENT_TABLE_NAME, con=engine, if_exists='append', index=False)
        print(f"  -> 최종 저장: {len(df_combined)} 건 완료.")
        return True

    except Exception as e:
        print(f"  -> 최종 DB 저장 실패: {e}")
        return False


# =============================================================================
# DB 조회/업데이트 함수
# =============================================================================
def get_regions_to_fetch_from_db() -> list:
    """
    DB에서 수집 대상 지역 목록 조회

    Returns:
        list: [(시군구코드, 마지막수집일), ...] 형태의 리스트
    """
    engine = get_engine()

    try:
        with engine.connect() as conn:
            query = f"""
                SELECT sgg_code, rent_last_fetched_date 
                FROM {REGION_TABLE_NAME}
                WHERE rent_last_fetched_date >= '{OLDEST_DATE_YMD}'
                ORDER BY sgg_code ASC
            """
            df_regions = pd.read_sql(query, con=conn)
            return list(df_regions.itertuples(index=False, name=None))

    except Exception as e:
        print(f"[오류] DB에서 지역 목록 조회 실패: {e}")
        return []


def update_fetch_progress_in_db(region_code: str, date_ym: str) -> None:
    """
    수집 진행 상황을 DB에 업데이트

    Args:
        region_code: 시군구 코드
        date_ym: 수집 완료된 년월 (YYYYMM)
    """
    engine = get_engine()

    try:
        with engine.connect() as conn:
            query = text(f"""
                UPDATE {REGION_TABLE_NAME}
                SET rent_last_fetched_date = :date_ym
                WHERE sgg_code = :region_code
            """)
            conn.execute(query, {"date_ym": date_ym, "region_code": region_code})
            conn.commit()

    except Exception as e:
        print(f"[경고] 진행 상황 업데이트 실패 (region: {region_code}): {e}")


# =============================================================================
# 메인 실행 루프
# =============================================================================
def main_fetch_loop():
    """
    라운드 로빈 방식으로 전월세 데이터를 수집합니다.

    각 지역에 대해 가장 오래된 미수집 월부터 순차적으로 수집하며,
    API 호출 한도에 도달하거나 모든 데이터 수집이 완료되면 종료합니다.
    """
    call_count = 0
    oldest_date_dt = pd.to_datetime(OLDEST_DATE_YMD, format='%Y%m')

    # 법정동 코드 맵 로드
    code_map = get_bjdong_code_map()
    if not code_map:
        print("[치명적 오류] 법정동 코드 맵을 생성할 수 없어 종료합니다.")
        return

    print(f"--- [전월세] 라운드 로빈 데이터 수집 시작 ---")
    print(f"    프로젝트 루트: {PROJECT_ROOT}")
    print(f"    API 호출 한도: {API_CALL_LIMIT_PER_RUN}")

    try:
        while call_count < API_CALL_LIMIT_PER_RUN:
            print(f"\n--- [전월세] 새 라운드 시작 (현재 호출: {call_count}) ---")

            regions = get_regions_to_fetch_from_db()
            if not regions:
                print("수집할 지역이 없습니다.")
                break

            work_done_in_this_round = False

            for region_code, last_fetched_date_str in regions:
                if call_count >= API_CALL_LIMIT_PER_RUN:
                    break

                # 다음 수집 대상 월 계산
                date_to_fetch_dt = (
                    pd.to_datetime(last_fetched_date_str, format='%Y%m') -
                    pd.DateOffset(months=1)
                )

                if date_to_fetch_dt < oldest_date_dt:
                    continue

                work_done_in_this_round = True
                date_ym_str = date_to_fetch_dt.strftime('%Y%m')

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"수집: {region_code}-{date_ym_str} (호출 {call_count + 1})"
                )
                call_count += 1

                success = fetch_rent_data_and_save(region_code, date_ym_str, code_map)

                if success:
                    update_fetch_progress_in_db(region_code, date_ym_str)
                else:
                    print(f"  -> [경고] {region_code}-{date_ym_str} 처리 실패.")

                time.sleep(SLEEP_TIME_BETWEEN_CALLS)

            if call_count >= API_CALL_LIMIT_PER_RUN:
                break

            if not work_done_in_this_round:
                print("이번 라운드에서 수집할 데이터가 없습니다.")
                break

    except KeyboardInterrupt:
        print("\n[중단] 사용자에 의해 중지되었습니다.")

    print(f"\n--- [전월세] 수집 완료 (총 {call_count}회 호출) ---")


# =============================================================================
# 엔트리포인트
# =============================================================================
if __name__ == "__main__":
    main_fetch_loop()