import pandas as pd
import sys
import os

# [중요] 'scripts' 폴더에 있는 함수를 import 하기 위해 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 테스트 대상 함수 임포트
from scripts.fetch_trade_data import parse_trade_xml_to_df

# --- 테스트용 가짜 XML 데이터 ---

# 1. 정상 작동 케이스 (제공해주신 샘플 기반)
SAMPLE_XML_OK = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<response>
    <header>
        <resultCode>00</resultCode>
        <resultMsg>OK</resultMsg>
    </header>
    <body>
        <items>
            <item>
                <aptNm>인왕산2차아이파크</aptNm>
                <dealAmount>63,400</dealAmount>
                <dealDay>22</dealDay>
                <dealMonth>12</dealMonth>
                <dealYear>2015</dealYear>
                <jibun>88</jibun>
                <sggCd>11110</sggCd>
                <umdNm>무악동</umdNm>
            </item>
            <item>
                <aptNm>종로센트레빌</aptNm>
                <dealAmount>52,300</dealAmount>
                <dealDay>26</dealDay>
                <dealMonth>12</dealMonth>
                <dealYear>2015</dealYear>
                <jibun>2-1</jibun>
                <sggCd>11110</sggCd>
                <umdNm>숭인동</umdNm>
            </item>
        </items>
    </body>
</response>
"""

# 2. API 에러 케이스
SAMPLE_XML_API_ERROR = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<response>
    <header>
        <resultCode>99</resultCode>
        <resultMsg>SERVICE_KEY_ERROR</resultMsg>
    </header>
</response>
"""

# 3. 데이터가 없는 케이스
SAMPLE_XML_EMPTY = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<response>
    <header>
        <resultCode>00</resultCode>
        <resultMsg>OK</resultMsg>
    </header>
    <body>
        <items></items>
        <totalCount>0</totalCount>
    </body>
</response>
"""


# --- 테스트 함수 정의 (함수명은 'test_'로 시작해야 함) ---

def test_parse_normal_case():
    """정상 XML이 올바르게 DataFrame으로 변환되는지 테스트"""
    df = parse_trade_xml_to_df(SAMPLE_XML_OK)

    # 1. 반환된 것이 DataFrame이 맞는지 확인
    assert isinstance(df, pd.DataFrame)

    # 2. 아이템 2개가 모두 파싱되었는지 확인
    assert len(df) == 2

    # 3. [중요] 특정 값 파싱이 정확한지 확인 (e.g., '2-1' -> 본번 '2', 부번 '1')
    item_2 = df.iloc[1]  # '종로센트레빌' 데이터
    assert item_2['시군구'] == '11110'
    assert item_2['법정동'] == '숭인동'
    assert item_2['본번'] == '2'
    assert item_2['부번'] == '1'
    assert item_2['거래금액(만원)'] == '52300'  # 콤마 제거 확인
    assert item_2['계약일'] == '20151226'  # 날짜 조합 확인


def test_parse_jibun_single():
    """본번만 있는 'jibun' (e.g., '88')이 올바르게 파싱되는지 테스트"""
    df = parse_trade_xml_to_df(SAMPLE_XML_OK)

    item_1 = df.iloc[0]  # '인왕산2차아이파크' 데이터
    assert item_1['본번'] == '88'
    assert item_1['부번'] == '0'  # 부번이 0으로 처리되었는지 확인


def test_parse_api_error_case():
    """API 에러(resultCode=99) 발생 시 빈 DataFrame을 반환하는지 테스트"""
    df = parse_trade_xml_to_df(SAMPLE_XML_API_ERROR)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_parse_empty_items_case():
    """정상 응답이지만 데이터가 0건일 때 빈 DataFrame을 반환하는지 테스트"""
    df = parse_trade_xml_to_df(SAMPLE_XML_EMPTY)
    assert isinstance(df, pd.DataFrame)
    assert df.empty