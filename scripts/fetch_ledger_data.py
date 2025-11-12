import requests, pandas as pd, xml.etree.ElementTree as ET
import os, sys, time
from datetime import datetime
from sqlalchemy import text
from calendar import monthrange

# [추가] 프로젝트 루트 경로를 sys.path에 추가
# (app.core.config를 찾기 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# [추가] 중앙 설정 파일에서 'engine'을 import
# (이 import 시점에 config.py가 실행되며 APP_ENV에 맞춰 engine이 생성됨)
from app.core.config import engine, load_dotenv

# --- 환경 변수 및 상수 설정 ---
load_dotenv()
API_SERVICE_KEY = os.getenv("API_SERVICE_KEY")
API_URL_LEDGER = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
LEDGER_TABLE_NAME = "raw_ledger" # 저장할 테이블
REGION_TABLE_NAME = "meta_bjdong_codes"

# [신규] 수집할 가장 *과거* 날짜 (경계)
OLDEST_DATE_YMD = "202301"

# --- API 제한 설정 (동일) ---
API_CALL_LIMIT_PER_RUN = 50
SLEEP_TIME_BETWEEN_CALLS = 0.5


# --- [신규] 1. 건축물대장 XML 파싱 함수 ---
def parse_ledger_xml_to_df(xml_text: str) -> pd.DataFrame:
    """건축물대장 표제부 API의 XML 응답을 DataFrame으로 파싱합니다."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  - XML 파싱 오류: {e}")
        return pd.DataFrame()

    result_code = root.findtext('.//resultCode', '99')
    if result_code != '00' and result_code != '000':
        result_msg = root.findtext('.//resultMsg', 'Unknown Error')
        print(f"  - API 에러: {result_msg} (코드: {result_code})")
        return pd.DataFrame()

    total_count = int(root.findtext('.//totalCount', '0'))
    if total_count == 0:
        return pd.DataFrame()

    items = root.findall('.//item')
    if not items:
        print(f"  - [경고] totalCount는 {total_count}이지만 'item' 태그를 찾을 수 없습니다.")
        return pd.DataFrame()

    records = []
    for item in items:
        # [핵심] 모델에 필요한 '주용도' 추출
        main_purpose = item.findtext('mainPurpsCdNm', '').strip()

        # [경고] '위반건축물' 플래그(violBldYn)가 표제부 샘플에 없습니다.
        # API 명세서 재확인 필요. 우선 빈 값으로 저장
        viol_bld_yn = item.findtext('violBldYn', 'N').strip() or 'N'  # (N:정상, Y:위반)

        record = {
            '시군구': item.findtext('sigunguCd'),
            '법정동': item.findtext('bjdongCd'),
            '본번': item.findtext('bun', '0').zfill(4),
            '부번': item.findtext('ji', '0').zfill(4),
            '주용도': main_purpose,
            '위반건축물여부': viol_bld_yn,
            '사용승인일': item.findtext('useAprDay', '').strip()
        }
        records.append(record)
    return pd.DataFrame(records)


# --- 2. API 호출 및 저장 함수 ---
def fetch_ledger_data_and_save(sgg_cd: str, bjdong_cd: str, start_date: str, end_date: str) -> bool:
    """단일 API 호출 및 DB 저장 (건축물대장용)"""
    params = {
        'serviceKey': API_SERVICE_KEY,
        'sigunguCd': sgg_cd,
        'bjdongCd': bjdong_cd,
        'startDate': start_date,  # (YYYYMMDD)
        'endDate': end_date,  # (YYYYMMDD)
        'numOfRows': '1000'
    }

    try:
        response = requests.get(API_URL_LEDGER, params=params, timeout=30)
        response.raise_for_status()

        xml_response_text = response.text
        df_api_data = parse_ledger_xml_to_df(xml_response_text)

        if not df_api_data.empty:
            df_api_data.to_sql(
                LEDGER_TABLE_NAME,  # 'raw_ledger' 테이블에 저장
                con=engine,
                if_exists='append',
                index=False
            )
            print(f"  -> 성공: {len(df_api_data)} 건 저장됨.")
        else:
            print("  -> 데이터 없음 (0건).")

        return True
    except Exception as e:
        print(f"  -> API/DB 처리 실패: {e}")
        return False


# --- 3. DB 기반 진행 상황 관리 함수 (수정) ---
def get_regions_to_fetch_from_db() -> list:
    """[수정] DB의 'meta_bjdong_codes' 테이블에서 (sgg_code, bjdong_code, date) 튜플 리스트를 읽어옵니다."""
    try:
        with engine.connect() as conn:
            # [수정] 컬럼명 및 테이블명 변경
            query = f"""
                SELECT sgg_code, bjdong_code, ledger_last_fetched_date 
                FROM {REGION_TABLE_NAME}
                WHERE ledger_last_fetched_date >= '{OLDEST_DATE_YMD}'
                ORDER BY sgg_code ASC, bjdong_code ASC
            """
            df_regions = pd.read_sql(query, con=conn)
            # (sgg_code, bjdong_code, last_fetched_date) 튜플 리스트 반환
            return list(df_regions.itertuples(index=False, name=None))
    except Exception as e:
        print(f"[오류] DB에서 법정동 목록(건축물대장)을 불러오는 데 실패했습니다: {e}")
        return []


def update_fetch_progress_in_db(region_code: str, dong_code: str, date_ym: str):
    """ DB의 'meta_bjdong_codes' 테이블을 업데이트합니다."""
    try:
        with engine.connect() as conn:
            query = text(f"""
                UPDATE {REGION_TABLE_NAME}
                SET ledger_last_fetched_date = :date_ym
                WHERE sgg_code = :region_code AND bjdong_code = :dong_code
            """)
            conn.execute(query, {"date_ym": date_ym, "region_code": region_code, "dong_code": dong_code})
            conn.commit()
    except Exception as e:
        print(f"[경고] DB 진행 상황(건축물대장) 업데이트 실패 (region: {region_code}-{dong_code}): {e}")


# --- 4. 메인 실행 루프 (수정) ---
def main_fetch_loop():
    """'라운드 로빈' 방식으로 건축물대장 데이터를 수집합니다."""

    call_count = 0
    oldest_date_dt = pd.to_datetime(OLDEST_DATE_YMD, format='%Y%m')

    print(f"--- [건축물대장] DB 기반 [라운드 로빈] 데이터 수집을 시작합니다. ---")

    try:
        while call_count < API_CALL_LIMIT_PER_RUN:
            print(f"\n--- [건축물대장] 새 수집 라운드 시작 (현재 호출: {call_count}) ---")

            # [수정] '시군구' 목록이 아닌 '법정동' 목록을 DB에서 바로 가져옴
            regions_bjdong = get_regions_to_fetch_from_db()
            if not regions_bjdong:
                print("수집할 법정동 목록이 없습니다.")
                break

            print(f"대상 법정동 {len(regions_bjdong)}개 로드 완료.")
            work_done_in_this_round = False

            # [수정] 이중 루프 -> 단일 루프
            # (sgg_code, dong_code, last_fetched_date_str) 튜플을 순회
            for sgg_code, dong_code, last_fetched_date_str in regions_bjdong:

                if call_count >= API_CALL_LIMIT_PER_RUN:
                    print(f"\n[중단] API 일일 호출 제한 도달.")
                    break

                date_to_fetch_dt = (pd.to_datetime(last_fetched_date_str, format='%Y%m') -
                                    pd.DateOffset(months=1))

                if date_to_fetch_dt < oldest_date_dt:
                    continue  # 이 법정동은 이미 수집 완료

                work_done_in_this_round = True

                # 날짜 (YYYYMMDD로 변환)
                date_ym_str = date_to_fetch_dt.strftime('%Y%m')
                year = date_to_fetch_dt.year
                month = date_to_fetch_dt.month
                start_date_str = date_to_fetch_dt.strftime('%Y%m01')
                end_date_str = f"{year}{month:02d}{monthrange(year, month)[1]}"

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] 수집 시도: {sgg_code}-{dong_code} (날짜: {date_ym_str}) (호출 {call_count + 1})")
                call_count += 1

                success = fetch_ledger_data_and_save(
                    sgg_cd=sgg_code,
                    bjdong_cd=dong_code,
                    start_date=start_date_str,
                    end_date=end_date_str
                )

                if success:
                    update_fetch_progress_in_db(sgg_code, dong_code, date_ym_str)
                else:
                    print(f"  -> [경고] {sgg_code}-{dong_code} 처리 실패.")

                time.sleep(SLEEP_TIME_BETWEEN_CALLS)

            # --- (법정동 루프 종료) ---
            if call_count >= API_CALL_LIMIT_PER_RUN:
                print("API 호출 제한으로 현재 세션을 종료합니다.")
                break

            if not work_done_in_this_round:
                print("이번 라운드에서 수집할 데이터가 없었습니다.")
                break

    except KeyboardInterrupt:
        print("\n[중단] 사용자에 의해 스크립트가 중지되었습니다.")

    print("\n--- [건축물대장] 데이터 수집 세션이 완료되었습니다. ---")

if __name__ == "__main__":
    main_fetch_loop()