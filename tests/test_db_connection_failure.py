"""
DB 연결 실패 시 /predict API 동작 테스트

테스트 목표:
1. DB 연결 실패 시 500 에러 응답 반환 확인
2. 무한 로딩 방지 확인 (타임아웃 내 응답)
3. 에러 응답 형식 검증
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime
import io
import os

# 테스트 대상 모듈
from app.main import app
from app.core.exceptions import DatabaseConnectionError
from app.core.config import check_db_connection, is_db_available, reset_db_availability


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def client():
    """FastAPI 테스트 클라이언트"""
    return TestClient(app)


@pytest.fixture
def mock_ocr_results():
    """OCR 결과 모킹 데이터"""
    return {
        'ledger': {
            'address': '인천광역시 부평구 삼산동 167-15',
            'exclusive_area': 59.94,
            'main_use': '공동주택'
        },
        'registry': {
            'address': '인천광역시 부평구 삼산동 167-15',
            'owner_name': '홍길동'
        }
    }


@pytest.fixture
def sample_image_file():
    """테스트용 이미지 파일 생성"""
    # 1x1 픽셀 PNG 이미지 (최소 크기)
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    return io.BytesIO(png_bytes)


@pytest.fixture
def sample_pdf_file():
    """테스트용 PDF 파일 생성"""
    pdf_bytes = b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 2\n0000000000 65535 f \n0000000009 00000 n \ntrailer\n<<>>\nstartxref\n29\n%%EOF'
    return io.BytesIO(pdf_bytes)


# =============================================================================
# DB 연결 실패 테스트
# =============================================================================
class TestDatabaseConnectionFailure:
    """DB 연결 실패 시나리오 테스트"""

    def test_predict_returns_500_on_db_connection_failure(
        self,
        client,
        sample_image_file,
        sample_pdf_file
    ):
        """
        DB 연결 실패 시 500 에러와 올바른 응답 형식 반환 확인
        """
        # Given: DB 연결 실패 상황 모킹
        # 중요: patch 경로는 "import된 위치" 기준이어야 함
        # app.main에서 from app.services.predict_service import predict_risk_with_ocr 했으므로
        # app.main.predict_risk_with_ocr를 patch해야 함
        with patch('app.main.predict_risk_with_ocr') as mock_predict, \
             patch('app.main.extract_building_ledger') as mock_ledger, \
             patch('app.main.extract_real_estate_data') as mock_registry, \
             patch('app.main.validate_document_match') as mock_validate:

            # OCR 함수들은 정상 동작
            mock_ledger.return_value = {'address': '테스트 주소'}
            mock_registry.return_value = {'address': '테스트 주소'}
            mock_validate.return_value = (True, "일치", {'confidence': 0.95, 'errors': [], 'match_scores': {}})

            # predict_risk_with_ocr가 DatabaseConnectionError 발생
            mock_predict.side_effect = DatabaseConnectionError(
                "MySQL 서버에 연결할 수 없습니다"
            )

            # When: /predict API 호출
            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "인천광역시 부평구 삼산동 167-15"
                },
                files=[
                    ("ledger_files", ("test.png", sample_image_file, "image/png")),
                    ("registry_files", ("test.pdf", sample_pdf_file, "application/pdf"))
                ]
            )

        # Then: 500 에러 응답
        assert response.status_code == 500

        # 응답 형식 검증
        data = response.json()
        assert "meta" in data
        assert data["meta"]["code"] == 500
        assert data["meta"]["message"] == "서버 오류가 발생했습니다"
        assert "timestamp" in data["meta"]

        assert "errors" in data
        assert len(data["errors"]) > 0
        assert data["errors"][0]["field"] == "server"
        assert "분석 실패" in data["errors"][0]["message"]

    def test_predict_response_within_timeout(
        self,
        client,
        sample_image_file,
        sample_pdf_file
    ):
        """
        DB 연결 실패 시에도 타임아웃 내 응답 반환 (무한 로딩 방지)
        """
        import time

        # patch 경로는 import된 위치 기준
        with patch('app.main.predict_risk_with_ocr') as mock_predict, \
             patch('app.main.extract_building_ledger') as mock_ledger, \
             patch('app.main.extract_real_estate_data') as mock_registry, \
             patch('app.main.validate_document_match') as mock_validate:

            mock_ledger.return_value = {'address': '테스트'}
            mock_registry.return_value = {'address': '테스트'}
            mock_validate.return_value = (True, "", {'confidence': 1.0, 'errors': [], 'match_scores': {}})

            mock_predict.side_effect = DatabaseConnectionError(
                "Database connection failed"
            )

            # When: API 호출 (타임아웃 10초 설정)
            start_time = time.time()
            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "테스트 주소"
                },
                files=[
                    ("ledger_files", ("test.png", sample_image_file, "image/png")),
                    ("registry_files", ("test.pdf", sample_pdf_file, "application/pdf"))
                ],
                timeout=10.0
            )
            elapsed_time = time.time() - start_time

        # Then: 10초 내 응답
        assert elapsed_time < 10.0, f"응답 시간이 너무 깁니다: {elapsed_time:.2f}초"
        assert response.status_code == 500

    def test_error_response_format(self, client, sample_image_file, sample_pdf_file):
        """
        에러 응답 형식이 정확한지 확인
        """
        expected_format = {
            "meta": {
                "code": 500,
                "message": "서버 오류가 발생했습니다",
                "timestamp": "2026-02-03T14:30:45"  # 형식만 확인
            },
            "errors": [
                {
                    "field": "server",
                    "message": "분석 실패: Database connection failed"
                }
            ]
        }

        # patch 경로는 import된 위치 기준
        with patch('app.main.predict_risk_with_ocr') as mock_predict, \
             patch('app.main.extract_building_ledger', return_value={}), \
             patch('app.main.extract_real_estate_data', return_value={}):

            mock_predict.side_effect = DatabaseConnectionError(
                "Database connection failed"
            )

            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "테스트 주소"
                },
                files=[
                    ("ledger_files", ("test.png", sample_image_file, "image/png")),
                    ("registry_files", ("test.pdf", sample_pdf_file, "application/pdf"))
                ]
            )

        data = response.json()

        # 구조 검증
        assert "meta" in data
        assert "code" in data["meta"]
        assert "message" in data["meta"]
        assert "timestamp" in data["meta"]
        assert "errors" in data

        # timestamp 형식 검증 (ISO 8601)
        try:
            datetime.fromisoformat(data["meta"]["timestamp"])
        except ValueError:
            pytest.fail("timestamp가 ISO 8601 형식이 아닙니다")


# =============================================================================
# 예외 클래스 테스트
# =============================================================================
class TestDatabaseConnectionError:
    """DatabaseConnectionError 예외 클래스 테스트"""

    def test_exception_with_message(self):
        """메시지만 있는 예외"""
        exc = DatabaseConnectionError("연결 실패")
        assert str(exc) == "연결 실패"
        assert exc.original_error is None

    def test_exception_with_original_error(self):
        """원본 예외 포함"""
        original = ConnectionError("Connection refused")
        exc = DatabaseConnectionError("연결 실패", original_error=original)

        assert "연결 실패" in str(exc)
        assert "Connection refused" in str(exc)
        assert exc.original_error is original

    def test_exception_default_message(self):
        """기본 메시지 확인"""
        exc = DatabaseConnectionError()
        assert "데이터베이스 연결" in str(exc)


# =============================================================================
# Price Service DB 실패 테스트
# =============================================================================
class TestPriceServiceDBFailure:
    """Price Service의 DB 연결 실패 처리 테스트"""

    def test_get_trade_price_raises_on_connection_failure(self):
        """get_trade_price가 DB 연결 실패 시 예외 발생"""
        from app.services.price_service import get_trade_price

        with patch('app.services.price_service.is_db_available', return_value=False):
            with pytest.raises(DatabaseConnectionError):
                get_trade_price("1123510100100010001", 59.94)

    def test_get_public_price_raises_on_connection_failure(self):
        """get_public_price가 DB 연결 실패 시 예외 발생"""
        from app.services.price_service import get_public_price

        with patch('app.services.price_service.is_db_available', return_value=False):
            with pytest.raises(DatabaseConnectionError):
                get_public_price("1123510100100010001", 59.94)

    def test_estimate_market_price_raises_on_connection_failure(self):
        """estimate_market_price가 DB 연결 실패 시 예외 발생"""
        from app.services.price_service import estimate_market_price

        with patch('app.services.price_service.is_db_available', return_value=False), \
             patch('app.services.price_service.get_trade_price') as mock_trade:

            mock_trade.side_effect = DatabaseConnectionError("연결 실패")

            with pytest.raises(DatabaseConnectionError):
                estimate_market_price("1123510100100010001", 59.94)


# =============================================================================
# Health Check 테스트
# =============================================================================
class TestHealthCheck:
    """헬스체크 엔드포인트 테스트"""

    def test_health_check_with_db_connected(self, client):
        """DB 연결 시 헬스체크"""
        with patch('app.main.check_db_connection', return_value=True):
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Healthy"
        assert data["database"] == "connected"

    def test_health_check_with_db_disconnected(self, client):
        """DB 미연결 시 헬스체크"""
        with patch('app.main.check_db_connection', return_value=False):
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Healthy"
        assert data["database"] == "disconnected"

    def test_db_health_endpoint_connected(self, client):
        """/health/db 엔드포인트 - 연결됨"""
        with patch('app.main.check_db_connection', return_value=True):
            response = client.get("/health/db")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_db_health_endpoint_disconnected(self, client):
        """/health/db 엔드포인트 - 미연결"""
        with patch('app.main.check_db_connection', return_value=False):
            response = client.get("/health/db")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"


# =============================================================================
# Integration 테스트 (실제 DB 없이 동작 확인)
# =============================================================================
class TestPredictIntegration:
    """통합 테스트 - 전체 흐름"""

    def test_full_flow_with_db_error(
        self,
        client,
        sample_image_file,
        sample_pdf_file
    ):
        """
        전체 흐름 테스트 - DB 오류 발생 시

        1. 파일 업로드 → 성공
        2. OCR 처리 → 성공
        3. 문서 검증 → 성공
        4. 예측 (DB 조회) → 실패
        5. 500 에러 응답 → 확인
        """
        with patch('app.main.extract_building_ledger') as mock_ledger, \
             patch('app.main.extract_real_estate_data') as mock_registry, \
             patch('app.main.validate_document_match') as mock_validate, \
             patch('app.main.predict_risk_with_ocr') as mock_predict:

            # OCR 성공
            mock_ledger.return_value = {'address': '인천광역시 부평구 삼산동 167-15'}
            mock_registry.return_value = {'address': '인천광역시 부평구 삼산동 167-15'}

            # 문서 검증 성공
            mock_validate.return_value = (True, "매칭 성공", {
                'confidence': 0.95,
                'errors': [],
                'match_scores': {'address': 1.0}
            })

            # 예측 단계에서 DB 연결 오류
            mock_predict.side_effect = DatabaseConnectionError(
                "Database connection failed"
            )

            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "인천광역시 부평구 삼산동 167-15"
                },
                files=[
                    ("ledger_files", ("test.png", sample_image_file, "image/png")),
                    ("registry_files", ("test.pdf", sample_pdf_file, "application/pdf"))
                ]
            )

        # 검증
        assert response.status_code == 500
        data = response.json()
        assert data["meta"]["code"] == 500
        assert "Database connection failed" in data["errors"][0]["message"]


# =============================================================================
# conftest.py 내용 (별도 파일로 분리 가능)
# =============================================================================
@pytest.fixture(autouse=True)
def reset_db_state():
    """각 테스트 전후로 DB 상태 초기화"""
    reset_db_availability()
    yield
    reset_db_availability()


# =============================================================================
# 실행
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])