import requests, pandas as pd, xml.etree.ElementTree as ET
import os, sys, time
from datetime import datetime
from sqlalchemy import text

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
API_URL_TRADE = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
TRADE_TABLE_NAME = "raw_trade" # 저장할 테이블
REGION_TABLE_NAME = "meta_region_codes"

# [신규] 수집할 가장 *과거* 날짜 (경계)
OLDEST_DATE_YMD = "202301"

# --- API 제한 설정 (동일) ---
API_CALL_LIMIT_PER_RUN = 10
SLEEP_TIME_BETWEEN_CALLS = 0.5

def parse_trade_xml_to_df(xml_text: str) -> pd.DataFrame:
    """매매 실거래가 API의 XML 응답을 DataFrame으로 파싱합니다."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f" - XML 파싱 오류: {e}")
        # 응답이 <response> 태그로 시작하지 않는 완전한 에러일 수 있음.
        print(f" - 응답 내용 (앞 200자): {xml_text[:200]}")
        return pd.DataFrame()

    # resultCode가 '00' (정상)이 아니면 에러 메시지 출력
    result_code = root.findtext('.//resultCode', '99')
    if result_code != '00' and result_code != '000':  # '00' 또는 '000'을 정상으로 간주
        result_msg = root.findtext('.//resultMsg', 'Unknown Error')
        print(f"  - API가 에러를 반환했습니다: {result_msg} (코드: {result_code})")
        return pd.DataFrame()

    items = root.findall('.//item')

    if not items:
        print("  - 'item' 태그를 찾을 수 없습니다.")
        return pd.DataFrame()

    records = []

    # 리스트를 순회하며 데이터 가공
    for item in items:
        # 1. 'jibun' 컬럼에서 '본번', '부번' 분리 (예: '2-1' -> 2, 1 / '88' -> 88, 0)
        bonbeon, bubeon = '0', '0'
        jibun_str = item.findtext('jibun', '0').strip()
        if jibun_str:
            parts = jibun_str.split('-')
            bonbeon = parts[0].lstrip('0') or '0'  # '0088' -> '88'
            if len(parts) > 1:
                bubeon = parts[1].lstrip('0') or '0'  # '01' -> '1'

        # 2. 날짜(Year, Month, Day) 조합
        deal_date = (
            f"{item.findtext('dealYear', '')}"
            f"{item.findtext('dealMonth', '').zfill(2)}"
            f"{item.findtext('dealDay', '').zfill(2)}"
        )

        # 3. 'dealAmount'의 콤마(,) 제거
        deal_amount_str = item.findtext('dealAmount', '0').replace(',', '').strip()

        # [수정] data_processor.py의 SQL 쿼리와 컬럼명 맞추기
        record = {
            '시군구': item.findtext('sggCd'),  # 'sggCd' (시군구 코드)
            '법정동': item.findtext('umdNm'),  # 'umdNm' (읍면동 이름)
            '본번': bonbeon,
            '부번': bubeon,
            '거래금액(만원)': deal_amount_str,
            '계약일': deal_date
        }
        records.append(record)

    return pd.DataFrame(records)


# --- 2. API 호출 및 저장 함수 ---
def fetch_trade_data_and_save(lawd_cd: str, deal_ymd: str) -> bool:
    """
    단일 지역(lawd_cd)과 날짜(deal_ymd)에 대해 API를 호출하고,
    파싱하여 DB에 저장한 뒤, 성공 여부(True/False)를 반환합니다.
    """

    # 1. API 요청 파라미터 설정
    params = {
        'serviceKey': API_SERVICE_KEY,
        'LAWD_CD': lawd_cd,  # 동적으로 전달받은 지역 코드
        'DEAL_YMD': deal_ymd,  # 동적으로 전달받은 거래 연월
        'numOfRows': '1000'  # (한 번에 1000개까지)
    }

    try:
        # 2. API 호출
        response = requests.get(API_URL_TRADE, params=params, timeout=30)
        # 2xx (정상) 상태 코드가 아니면 예외 발생
        response.raise_for_status()

        # 3. XML 텍스트를 파싱 함수로 전달
        xml_response_text = response.text
        df_api_data = parse_trade_xml_to_df(xml_response_text)

        # 4. 파싱 결과가 있을 경우에만 DB에 저장
        if not df_api_data.empty:
            df_api_data.to_sql(
                TRADE_TABLE_NAME,
                con=engine,
                if_exists='append',  # 기존 테이블에 데이터 추가
                index=False
            )
            print(f"  -> 성공: {len(df_api_data)} 건 저장됨.")
        else:
            # (데이터가 0건인 것도 API 호출 '성공'으로 간주)
            print("  -> 데이터 없음 (0건).")

        # API 호출 및 처리가 성공적으로 완료됨
        return True

    except requests.exceptions.RequestException as e:
        # (네트워크 오류, 타임아웃, 4xx/5xx 에러 등)
        print(f"  -> API 호출 실패: {e}")
        return False  # API 호출 실패
    except Exception as e:
        # (XML 파싱 오류, DB 저장 오류 등)
        print(f"  -> 알 수 없는 오류 (파싱 또는 DB 저장): {e}")
        return False  # 기타 실패


# --- 3. DB 기반 진행 상황 관리 함수 (수정) ---
def get_regions_to_fetch_from_db() -> list:
    """DB에서 수집할 (시군구코드, 마지막수집일) 튜플 리스트를 읽어옵니다."""
    try:
        with engine.connect() as conn:
            # [수정] OLDEST_DATE_YMD보다 *크거나 같은* 날짜만 수집 대상으로 함
            query = f"""
                SELECT sgg_code, trade_last_fetched_date 
                FROM {REGION_TABLE_NAME}
                WHERE trade_last_fetched_date >= '{OLDEST_DATE_YMD}'
                ORDER BY sgg_code ASC
            """
            df_regions = pd.read_sql(query, con=conn)
            return list(df_regions.itertuples(index=False, name=None))
    except Exception as e:
        print(f"[오류] DB에서 지역 목록을 불러오는 데 실패했습니다: {e}")
        return []


def update_fetch_progress_in_db(region_code: str, date_ym: str):
    """DB의 last_fetched_date를 *현재 성공한 날짜*로 업데이트합니다."""
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
        print(f"[경고] DB 진행 상황 업데이트 실패 (region: {region_code}): {e}")


# --- 4. 메인 실행 루프 (수정) ---
def main_fetch_loop():
    """
    [수정] "Round-Robin" (Breadth-First) 방식으로 데이터를 수집합니다.
    API 호출 제한에 도달할 때까지 *모든 지역*을 순회하며 *1개월치*씩 수집합니다.
    """

    call_count = 0
    oldest_date_dt = pd.to_datetime(OLDEST_DATE_YMD, format='%Y%m')

    print(f"--- 'raw_trade' DB 기반 [라운드 로빈] 데이터 수집을 시작합니다. ---")

    try:
        # [수정] API 호출 제한에 도달할 때까지 전체 프로세스를 반복
        while call_count < API_CALL_LIMIT_PER_RUN:

            print(f"\n--- 새 수집 라운드 시작 (현재 호출: {call_count}/{API_CALL_LIMIT_PER_RUN}) ---")

            # 1. DB에서 *현재* 수집해야 할 지역 목록을 *매번* 새로고침
            regions = get_regions_to_fetch_from_db()
            if not regions:
                print("수집할 지역 목록이 없습니다. (모든 지역이 OLDEST_DATE_YMD에 도달)")
                break  # while True 루프 종료

            print(f"대상 지역 {len(regions)}개 로드 완료.")

            # 2. [신규] 이번 라운드에서 *실제로 작업이 수행되었는지* 확인하는 플래그
            work_done_in_this_round = False

            # [수정] 'for' 루프 (지역별)
            for region_code, last_fetched_date_str in regions:

                # 3. API 호출 제한 확인 (내부 루프)
                if call_count >= API_CALL_LIMIT_PER_RUN:
                    print(f"\n[중단] API 일일 호출 제한({API_CALL_LIMIT_PER_RUN}회)에 도달.")
                    break  # for 루프 중단

                # 4. 수집할 날짜 계산 (한 달 전)
                date_to_fetch_dt = (pd.to_datetime(last_fetched_date_str, format='%Y%m') -
                                    pd.DateOffset(months=1))

                # 5. [신규] 이 지역이 이미 수집 완료되었는지 확인
                if date_to_fetch_dt < oldest_date_dt:
                    # (이미 '201501'까지 수집했다면, last_fetched_date가 '201501'임)
                    # (date_to_fetch_dt는 '201412'가 되므로, 스킵)
                    continue  # 다음 지역으로

                # 6. (수집할 작업이 있었음)
                work_done_in_this_round = True
                date_ym_str = date_to_fetch_dt.strftime('%Y%m')

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] 수집 시도: {region_code}-{date_ym_str} (호출 {call_count + 1})")
                call_count += 1

                # 7. 실제 API 호출
                success = fetch_trade_data_and_save(region_code, date_ym_str)

                if success:
                    # 8. 성공 시 DB 상태를 *현재 날짜*로 업데이트
                    update_fetch_progress_in_db(region_code, date_ym_str)
                else:
                    print(f"  -> [경고] {region_code}-{date_ym_str} 처리 실패. (다음 라운드에서 재시도)")

                time.sleep(SLEEP_TIME_BETWEEN_CALLS)

            # --- (for 루프 종료) ---

            # 9. API 제한으로 중단되었거나, 모든 작업이 완료되었으면 while 루프 종료
            if call_count >= API_CALL_LIMIT_PER_RUN:
                print("API 호출 제한에 도달하여 현재 세션을 종료합니다.")
                break  # while 루프 종료

            if not work_done_in_this_round:
                print("이번 라운드에서 수집할 데이터가 없었습니다. (모든 지역 최신화 완료)")
                break  # while 루프 종료

    except KeyboardInterrupt:
        print("\n[중단] 사용자에 의해 스크립트가 중지되었습니다. (DB에 진행 상황 저장됨)")

    print("\n--- 데이터 수집 세션이 완료되었습니다. ---")

if __name__ == "__main__":
    main_fetch_loop()