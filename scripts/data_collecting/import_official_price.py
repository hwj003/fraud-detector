import pandas as pd
import os
import sys

# 프로젝트 설정 (db_manager 위치에 맞게 수정)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.database import engine # engine 가져오기


def import_csv_to_db(csv_path):
    print(f"[Start] CSV 로딩 시작: {csv_path}")

    # 1. CSV 읽기 (레이아웃 컬럼명 매핑)
    # 실제 CSV 파일의 헤더가 없는 경우 names=... 옵션 사용 필요할 수 있음
    # 여기서는 헤더가 있다고 가정 (레이아웃 기준 영문 매핑)
    try:
        # 컬럼 타입 지정 (문자열이 0으로 시작하는 경우 짤림 방지)
        dtype_map = {
            '법정동코드': str, '특수지코드': str,
            '본번': str, '부번': str,
            '동명': str, '호명': str, '기준연도': str
        }
        df = pd.read_csv(csv_path, encoding='utf-8', dtype=dtype_map)
        # 주의: 공공데이터는 보통 cp949 또는 euc-kr 인코딩입니다.
    except Exception as e:
        print(f"[Error] 파일 읽기 실패: {e}")
        return

    print(f"[Process] 데이터 가공 중... ({len(df)}건)")

    # 2. PNU 생성 (가장 중요)
    # 레이아웃 기준: 법정동코드(10) + 특수지코드(1) + 본번(4) + 부번(4)
    # 예: 1117012300 + 0 + 0048 + 0000

    # 본번/부번 4자리 패딩 (zfill)
    df['본번'] = df['본번'].fillna('0').astype(str).str.zfill(4)
    df['부번'] = df['부번'].fillna('0').astype(str).str.zfill(4)
    df['특수지코드'] = df['특수지코드'].fillna('0').astype(str)

    # PNU 컬럼 생성
    df['pnu'] = df['법정동코드'] + df['특수지코드'] + df['본번'] + df['부번']

    # 3. DB 적재를 위한 컬럼명 변경 (DataFrame -> Table)
    rename_map = {
        '기준연도': 'base_year',
        '동명': 'dong_name',
        '호명': 'ho_name',
        '공시가격': 'price',
        '전용면적': 'exclusive_area',
        '단지명': 'complex_name',
        '도로명주소': 'road_address'
    }

    # 필요한 컬럼만 선택 및 이름 변경
    export_df = df.rename(columns=rename_map)

    # 시군구/법정동 분리
    export_df['sigungu_code'] = export_df['법정동코드'].str[:5]
    export_df['bjdong_code'] = export_df['법정동코드'].str[5:]

    # 최종 저장할 컬럼들
    target_cols = [
        'pnu', 'sigungu_code', 'bjdong_code',
        'dong_name', 'ho_name',
        'price', 'exclusive_area', 'base_year',
        'complex_name', 'road_address'
    ]

    export_df = export_df[target_cols]

    # 4. 고속 적재 (pandas to_sql 사용)
    print("[DB] 데이터 저장 시작 (시간이 좀 걸릴 수 있습니다)...")
    try:
        # if_exists='append': 데이터 추가 모드 (replace하면 테이블 날아감 주의)
        # chunksize: 메모리 터짐 방지
        export_df.to_sql('official_price_raw', con=engine, if_exists='append', index=False, chunksize=10000)
        print(f"[Success] 총 {len(export_df)}건 저장 완료!")
    except Exception as e:
        print(f"[Error] DB 저장 실패: {e}")


if __name__ == "__main__":

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    CSV_PATH = os.path.join(project_root, 'data', '국토교통부_주택 공시가격 정보(2024)_수정.csv')
    # 여기에 다운로드 받은 CSV 파일 경로 입력

    if os.path.exists(CSV_PATH):
        import_csv_to_db(CSV_PATH)
    else:
        print("CSV 파일이 없습니다.")