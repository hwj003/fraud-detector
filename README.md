# 전세사기 예측 모델 
### API 엔드포인트
/predict - post 
### 입력 파라미터 (json body)
address: 분석할 매물의 도로명 또는 지번 주소 (예: '인천광역시 부평구 산곡동 145') \
deposit: 계약하려는 전세 보증금 액수 (단위: 만원, 예: 20000)
### 입력 예시
{ \
    "address": 인천광역시 부평구 산곡동 145, \
    "deposit": 20000\
} 
### 출력 예시
{
    "address": "인천광역시 부평구 산곡동 145", // 주소 \
    "building_name": "311동 101호", // 건축물대장상 건물명 및 동/호수 \
    "deposit": "18000만원", // 전세 보증금 \
    "risk_score": 42.0, // AI 모델이 예측한 전세사기 위험 확률 (0 ~ 100점) \
    "risk_level": "SAFE", // 위험 등급 (SAFE: 안전, RISKY: 위험)\
    "details": { \
        "hug_ratio": 86.1, // HUG 전세보증보험 가입 한도 대비 전세가 비율 (%, 100% 초과 시 가입 불가) \
        "total_ratio": 97.5, // 깡통전세 위험도: (추정 대출금 + 전세금) / 추정 시세 비율 (%, 100% 초과 시 위험)\
        "is_trust": false, // 신탁 등기 여부 (True: 신탁사 소유로 주의 필요, False: 일반 매물)\
        "is_short_term": false // 집주인 단기 소유 여부 (True: 소유권 이전 2년 미만, False: 장기 보유)\
    } \
}

공공데이터 api 신청 및 인증 키 필요 (.env에 추가) 
### 사용 데이터 목록 
##### 국토교통부_오피스텔 전월세 실거래가 자료  https://www.data.go.kr/data/15126475/openapi.do 
##### 국토교통부_오피스텔 매매 실거래가 자료 https://www.data.go.kr/data/15126464/openapi.do
##### 국토교통부_연립다세대 전월세 실거래가 자료 https://www.data.go.kr/data/15126473/openapi.do
##### 국토교통부_연립다세대 매매 실거래가 자료 https://www.data.go.kr/data/15126467/openapi.do
##### 국토교통부_아파트 매매 실거래가 자료 https://www.data.go.kr/data/15126469/openapi.do
##### 국토교통부_아파트 전월세 실거래가 자료 https://www.data.go.kr/data/15126474/openapi.do
##### 세움터 집합 건축물대장 (표제부/전유부)


### 사용 방법 (실행순서 중요) 
1. pip install -r requirements.txt  라이브러리 설치
2. python setup_region_codes.py 법정동코드 추가
3. python fetch_trade_data.py 실거래가 데이터 수집 
4. python fetch_rent_data.py 전월세 데이터 수집
5. python train_model.py 모델 훈련
6. python run_api.py 실행 (1~5번과정은 한 번만 실행하면 됨.)

 
