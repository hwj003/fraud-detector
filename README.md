### api 엔드포인트
/predict - post 
### 입력 파라미터 (json body)
jeonse_ratio: 전세가율 (0.0~1.0 사이의 값) \
is_illegal_building: 불법 건축물 여부 (0 아니면 1의 값) \
building_age: 건물 나이 (float 값) \
building_use: 건물의 주요 용도 (아파트, 다세대주택, 오피스텔 등) \
has_trust: 신탁 등기 여부 (0 아니면 1의 값) \
loan_plus_jeonse_ratio: 부채+전세가율(선순위 부채와 전세 보증금의 합이 매매가에서 차지하는 비율) (float 값)

### 입력 예시
{ \
    "jeonse_ratio": 0.7, \
    "is_illegal_building": 1, \
    "building_age": 8.0, \
    "building_use": "오피스텔", \
    "has_trust": 0, \
    "loan_plus_jeonse_ratio": 0.85 \
} 
### 출력 예시
{ \
    "risk_level": "RISKY", \
    "probability": 0.6353 \
} 

공공데이터 api 신청 및 인증 키 필요 (.env에 추가) 
### 사용 데이터 목록 
##### 국토교통부_오피스텔 전월세 실거래가 자료  https://www.data.go.kr/data/15126475/openapi.do 
##### 국토교통부_오피스텔 매매 실거래가 자료 https://www.data.go.kr/data/15126464/openapi.do
##### 국토교통부_연립다세대 전월세 실거래가 자료 https://www.data.go.kr/data/15126473/openapi.do
##### 국토교통부_연립다세대 매매 실거래가 자료 https://www.data.go.kr/data/15126467/openapi.do
##### 국토교통부_아파트 매매 실거래가 자료 \ https://www.data.go.kr/data/15126469/openapi.do
##### 국토교통부_아파트 전월세 실거래가 자료 \ https://www.data.go.kr/data/15126474/openapi.do
##### 국토교통부_건축HUB_건축물대장정보 서비스 https://www.data.go.kr/data/15134735/openapi.do


### 사용 방법 (실행순서 중요) 
1. pip install -r requirements.txt  라이브러리 설치
2. python setup_region_codes.py 법정동코드 추가
3. python fetch_trade_data.py 실거래가 데이터 수집 
4. python fetch_rent_data.py 전월세 데이터 수집
5. python train_model.py 모델 훈련
6. python run_api.py 실행 (1~5번과정은 한 번만 실행하면 됨.)

 
