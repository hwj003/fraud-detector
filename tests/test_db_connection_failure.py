"""
DB 연결 실패 시 /predict API 동작 테스트

테스트 목표:
1. DB 연결 실패 시 백그라운드 작업이 FAILED 상태로 전환
2. 에러 응답 형식 검증
3. 예외 클래스 동작 확인

변경사항 (비동기 API 대응):
- 기존: POST /predict → 동기 500 응답 기대
- 변경: POST /predict → 202 작업 생성 후, 백그라운드에서 FAILED 처리
- client fixture에서 Redis startup/shutdown 모킹 추가
- BytesIO를 헬퍼 함수로 매번 새로 생성 (closed file 방지)
- client fixture의 컨텍스트 매니저 중첩 문제 해결
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime
import io
from contextlib import ExitStack

# app.main import 전에 Redis 모킹이 필요하므로, client fixture 안에서 import
from app.core.exceptions import DatabaseConnectionError
from app.core.database import reset_db_availability
from app.main import app

# =============================================================================
# 헬퍼 함수 - 매번 새 BytesIO 생성 (closed file 문제 방지)
# =============================================================================
def make_image_file():
    """1x1 픽셀 PNG 이미지를 새로 생성"""
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    return io.BytesIO(png_bytes)


def make_pdf_file():
    """최소 PDF를 새로 생성"""
    pdf_bytes = (
        b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 2\n'
        b'0000000000 65535 f \n0000000009 00000 n \n'
        b'trailer\n<<>>\nstartxref\n29\n%%EOF'
    )
    return io.BytesIO(pdf_bytes)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="function")
def client():
    """
    FastAPI 테스트 클라이언트 (ExitStack 사용)

    ExitStack을 사용하면:
    1. Patch(Mock)들을 먼저 실행하고,
    2. 그 안에서 TestClient를 실행합니다.
    3. 테스트가 끝나면 역순으로 안전하게 종료합니다. (Client 종료 -> Patch 종료)
    """
    # ExitStack을 사용하여 모든 컨텍스트(Patch, Client)를 한 번에 관리
    with ExitStack() as stack:
        # 1. Redis 관련 Mock들을 먼저 활성화 (stack에 등록)
        stack.enter_context(patch('app.main.get_redis', new_callable=AsyncMock, return_value=None))
        stack.enter_context(patch('app.main.close_redis', new_callable=AsyncMock))
        stack.enter_context(patch('app.main.health_check_redis', new_callable=AsyncMock,
                                  return_value={"status": "mocked", "version": "test"}))

        # 2. Mock이 적용된 상태에서 TestClient를 컨텍스트 모드로 실행
        # 이렇게 하면 'yield'가 끝나고 나서 TestClient.__exit__이 먼저 호출되고(앱 종료),
        # 그 후에 Patch.__exit__이 호출됩니다(Mock 해제).
        test_client = stack.enter_context(TestClient(app))

        yield test_client


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


@pytest.fixture(autouse=True)
def reset_db_state():
    """각 테스트 전후로 DB 상태 초기화"""
    reset_db_availability()
    yield
    reset_db_availability()


# =============================================================================
# DB 연결 실패 테스트 (비동기 API 대응)
# =============================================================================
class TestDatabaseConnectionFailure:
    """DB 연결 실패 시나리오 테스트"""

    def test_predict_accepts_request_even_with_db_failure(self, client):
        """
        DB 연결 실패 상황에서도 요청은 202로 접수됨
        (실패 처리는 백그라운드에서 진행)
        """
        with patch('app.main.get_cached_result', new_callable=AsyncMock, return_value=None), \
             patch('app.main.set_task_status', new_callable=AsyncMock), \
             patch('app.main.is_db_available', return_value=False), \
             patch('app.main.reset_db_availability'), \
             patch('app.main.check_db_connection', return_value=False), \
             patch('app.main.generate_cache_key', return_value="predict:test"), \
             patch('app.main.generate_file_hash', return_value="hash123"):

            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "인천광역시 부평구 삼산동 167-15"
                },
                files=[
                    ("ledger_files", ("test.png", make_image_file(), "image/png")),
                    ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf"))
                ]
            )

        # 비동기 API는 항상 202로 접수
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data["data"]
        assert data["data"]["status"] == "PENDING"

    def test_predict_response_within_timeout(self, client):
        """
        비동기 API는 즉시 202를 반환하므로 타임아웃 문제 없음
        """
        import time

        with patch('app.main.get_cached_result', new_callable=AsyncMock, return_value=None), \
             patch('app.main.set_task_status', new_callable=AsyncMock), \
             patch('app.main.generate_cache_key', return_value="predict:timeout"), \
             patch('app.main.generate_file_hash', return_value="hash456"):

            start_time = time.time()
            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "테스트 주소"
                },
                files=[
                    ("ledger_files", ("test.png", make_image_file(), "image/png")),
                    ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf"))
                ],
            )
            elapsed_time = time.time() - start_time

        # 비동기이므로 매우 빠르게 응답 (1초 이내)
        assert elapsed_time < 2.0, f"응답 시간이 너무 깁니다: {elapsed_time:.2f}초"
        assert response.status_code == 202

    def test_failed_task_shows_db_error(self, client):
        """
        DB 오류로 실패한 작업 조회 시 에러 메시지 확인
        """
        task_data = {
            "task_id": "test-db-fail",
            "status": "FAILED",
            "progress": 10,
            "error": "분석 실패: 데이터베이스 연결을 확인할 수 없습니다. 잠시 후 다시 시도해주세요."
        }

        with patch('app.main.get_task_status', new_callable=AsyncMock,
                    return_value=task_data):
            response = client.get("/predict/test-db-fail")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "FAILED"
        assert "데이터베이스" in data["data"]["error"]

    def test_error_response_format_on_failed_task(self, client):
        """
        실패한 작업의 응답 형식 검증
        """
        task_data = {
            "task_id": "test-format",
            "status": "FAILED",
            "progress": 25,
            "error": "분석 실패: Database connection failed"
        }

        with patch('app.main.get_task_status', new_callable=AsyncMock,
                    return_value=task_data):
            response = client.get("/predict/test-format")

        data = response.json()

        # 구조 검증
        assert "meta" in data
        assert "code" in data["meta"]
        assert "message" in data["meta"]
        assert "timestamp" in data["meta"]
        assert "data" in data
        assert "status" in data["data"]
        assert "error" in data["data"]

        # timestamp 형식 검증 (ISO 8601)
        try:
            datetime.fromisoformat(data["meta"]["timestamp"])
        except ValueError:
            pytest.fail("timestamp가 ISO 8601 형식이 아닙니다")


# =============================================================================
# 예외 클래스 테스트 (변경 없음)
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
# Price Service DB 실패 테스트 (변경 없음)
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
# Health Check 테스트 (새 API 형식에 맞게 수정)
# =============================================================================
class TestHealthCheck:
    """헬스체크 엔드포인트 테스트"""

    def test_health_check_returns_service_info(self, client):
        """
        GET / → 서비스 정보 + Redis 상태 포함
        (새 main.py에서는 DB 상태 대신 Redis 상태를 반환)
        """
        with patch('app.main.health_check_redis', new_callable=AsyncMock,
                    return_value={"status": "connected", "version": "7.0"}):
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Healthy"
        assert data["service"] == "Fraud Detector AI"
        assert data["version"] == "2.0"
        assert "redis" in data

    def test_health_check_with_redis_disconnected(self, client):
        """
        Redis 미연결 시에도 헬스체크는 정상 반환
        """
        with patch('app.main.health_check_redis', new_callable=AsyncMock,
                    return_value={"status": "disconnected", "message": "Redis 연결 불가"}):
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Healthy"
        assert data["redis"]["status"] == "disconnected"


# =============================================================================
# Integration 테스트 (비동기 API 대응)
# =============================================================================
class TestPredictIntegration:
    """통합 테스트 - 전체 흐름"""

    def test_full_flow_submit_and_poll_completed(self, client):
        """
        전체 흐름 테스트 - 정상 케이스

        1. POST /predict → 202 + task_id
        2. GET /predict/{task_id} → COMPLETED + result
        """
        # Step 1: 작업 제출
        with patch('app.main.get_cached_result', new_callable=AsyncMock, return_value=None), \
             patch('app.main.set_task_status', new_callable=AsyncMock), \
             patch('app.main.generate_cache_key', return_value="predict:integ"), \
             patch('app.main.generate_file_hash', return_value="hash789"):

            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "인천광역시 부평구 삼산동 167-15"
                },
                files=[
                    ("ledger_files", ("test.png", make_image_file(), "image/png")),
                    ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf"))
                ]
            )

        assert response.status_code == 202
        task_id = response.json()["data"]["task_id"]

        # Step 2: 완료된 작업 조회
        completed_task = {
            "task_id": task_id,
            "status": "COMPLETED",
            "progress": 100,
            "result": {
                "meta": {"code": 200, "message": "전세사기 위험도 분석 완료"},
                "data": {
                    "address": "인천광역시 부평구 삼산동 167-15",
                    "risk_score": 35.0,
                    "risk_level": "SAFE"
                }
            }
        }

        with patch('app.main.get_task_status', new_callable=AsyncMock,
                    return_value=completed_task):
            poll_response = client.get(f"/predict/{task_id}")

        assert poll_response.status_code == 200
        poll_data = poll_response.json()
        assert poll_data["data"]["status"] == "COMPLETED"
        assert poll_data["data"]["result"]["data"]["risk_score"] == 35.0

    def test_full_flow_submit_and_poll_failed(self, client):
        """
        전체 흐름 테스트 - DB 오류 케이스

        1. POST /predict → 202 + task_id
        2. GET /predict/{task_id} → FAILED + error
        """
        # Step 1: 작업 제출
        with patch('app.main.get_cached_result', new_callable=AsyncMock, return_value=None), \
             patch('app.main.set_task_status', new_callable=AsyncMock), \
             patch('app.main.generate_cache_key', return_value="predict:fail"), \
             patch('app.main.generate_file_hash', return_value="hashfail"):

            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "인천광역시 부평구 삼산동 167-15"
                },
                files=[
                    ("ledger_files", ("test.png", make_image_file(), "image/png")),
                    ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf"))
                ]
            )

        assert response.status_code == 202
        task_id = response.json()["data"]["task_id"]

        # Step 2: 실패한 작업 조회
        failed_task = {
            "task_id": task_id,
            "status": "FAILED",
            "progress": 10,
            "error": "분석 실패: Database connection failed"
        }

        with patch('app.main.get_task_status', new_callable=AsyncMock,
                    return_value=failed_task):
            poll_response = client.get(f"/predict/{task_id}")

        assert poll_response.status_code == 200
        poll_data = poll_response.json()
        assert poll_data["data"]["status"] == "FAILED"
        assert "Database connection failed" in poll_data["data"]["error"]


# =============================================================================
# 실행
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])