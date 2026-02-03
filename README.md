# 🏠 전세사기 위험도 탐지 시스템 (Fraud Detector)

전세 계약 시 사기 피해를 예방하기 위한 AI 기반 위험도 분석 서비스입니다. 건축물대장과 등기부등본을 OCR로 분석하고, 실거래가/공시지가 데이터를 활용하여 종합적인 전세사기 위험도를 산출합니다.

---

## 📌 주요 기능

### 1. OCR 기반 문서 분석
- **건축물대장 분석**: Gemini API를 활용한 이미지 OCR
- **등기부등본 분석**: PDF 파싱 및 권리관계 추출
- **문서 매칭 검증**: 두 문서가 동일 물건인지 자동 검증

### 2. 시세 조회 및 분석
- 국토교통부 실거래가 API 연동 (매매/전월세)
- 공시지가 데이터 활용
- 전세가율 자동 계산

### 3. AI 위험도 예측
- Random Forest 모델 기반 위험도 점수 산출
- 다중 위험 요인 분석:
  - 전세가율 (LTV)
  - 선순위 채권 금액
  - 신탁 등기 여부
  - 위반 건축물 여부
  - 소유 기간 (깡통전세 위험)
  - 건물 노후도

### 4. HUG 전세보증보험 진단
- 보증 가입 가능 여부 판단
- 안전 보증 한도액 계산

### 5. 지역별 통계 시각화
- 지역별 전세가율 현황
- 위험 등급별 거래 분포
- 시계열 추이 분석

---

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| **Backend** | FastAPI, Python 3.10+ |
| **Database** | MySQL (SQLAlchemy ORM) |
| **AI/ML** | scikit-learn, Random Forest |
| **OCR** | Google Gemini API |
| **외부 API** | 국토교통부 실거래가 API, 카카오 로컬 API |
| **인프라** | Docker (선택) |

---

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/fraud-detector.git
cd fraud-detector

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 항목을 설정합니다:

```env
# 데이터베이스
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=fraud_detector

# 외부 API 키
API_SERVICE_KEY=국토교통부_API_키
KAKAO_REST_API_KEY=카카오_API_키
GEMINI_API_KEY=구글_Gemini_API_키

# 앱 설정
APP_ENV=local
```

### 3. 데이터베이스 초기화

```bash
# 테이블 생성 (자동)
python -c "from app.core import init_db; init_db()"

# 법정동 코드 등록
python scripts/setup_region_codes.py
```

### 4. 서버 실행

```bash
# 개발 모드
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 또는
python app/main.py
```

서버 실행 후 API 문서 확인: `http://localhost:8000/docs`

---

## 📡 API 엔드포인트

### 위험도 분석

```http
POST /predict
Content-Type: multipart/form-data
```

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `deposit` | int | ✅ | 보증금 (만원) |
| `address` | string | ✅ | 주소 (시세 조회용) |
| `ledger_files` | file[] | ✅ | 건축물대장 이미지 (PNG/JPG, 최대 5개) |
| `registry_files` | file[] | ✅ | 등기부등본 PDF (최대 3개) |

#### 응답 예시

```json
{
  "meta": {
    "code": 200,
    "message": "전세사기 위험도 분석 완료",
    "timestamp": "2026-02-04T14:30:45"
  },
  "data": {
    "address": "인천광역시 부평구 삼산동 167-15",
    "deposit": 35000000,
    "market_price": 61000000,
    "price_source": "DB_Trade",
    "risk_score": 41.0,
    "risk_level": "SAFE",
    "major_risk_factors": [
      {
        "type": "HIGH_LTV",
        "severity": "MEDIUM",
        "message": "전세가율이 다소 높습니다 (57.4%)"
      }
    ],
    "hug_result": {
      "is_eligible": true,
      "safe_limit": 43260000,
      "coverage_ratio": 100.0,
      "message": "HUG 전세보증보험 가입 가능"
    },
    "details": {
      "jeonse_ratio": 57.4,
      "senior_debt": 0,
      "is_illegal_building": false,
      "is_trust": false,
      "building_age": 15.3
    },
    "recommendations": [
      "HUG 전세보증보험 가입을 권장합니다",
      "전입신고 및 확정일자를 반드시 받으세요"
    ]
  }
}
```

### 상태 확인

```http
GET /
```

---

## 📊 데이터 수집

### 실거래가 데이터 수집

```bash
# 전월세 실거래가
python -m scripts.fetch_data.fetch_rent_data

# 매매 실거래가
python -m scripts.fetch_data.fetch_trade_data
```

### 모델 학습

```bash
python scripts/train_model.py
```

---

## 🔒 위험 등급 기준

| 등급 | 점수 범위 | 설명 |
|------|-----------|------|
| 🟢 SAFE | 0 ~ 40 | 안전 |
| 🟡 CAUTION | 41 ~ 70 | 주의 필요 |
| 🔴 RISKY | 71 ~ 100 | 위험 |

### 주요 위험 요인

| 요인 | 설명 |
|------|------|
| `HIGH_LTV` | 전세가율 80% 이상 |
| `SENIOR_DEBT` | 선순위 채권 존재 |
| `TRUST_PROPERTY` | 신탁 등기된 부동산 |
| `ILLEGAL_BUILDING` | 위반 건축물 |
| `OWNERSHIP_PERIOD` | 소유 기간 6개월 미만 |
| `OLD_BUILDING` | 건물 노후도 30년 이상 |

---