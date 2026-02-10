"""
비동기 /predict API 테스트

테스트 항목:
1. 비동기 작업 생성 (202 Accepted)
2. 캐시 히트 시 즉시 응답 (200 OK)
3. 작업 상태 폴링 (/predict/{task_id})
4. DB 연결 실패 시 적절한 에러
5. 캐시 무효화 (/predict/cache/{cache_key})
6. 입력 검증 (파일 누락 등)

수정사항:
- BytesIO fixture → 헬퍼 함수로 매번 새로 생성 (closed file 방지)
- mock 패치 경로를 main.py의 import 기준으로 통일 (app.main.xxx)
- client fixture에서 startup/shutdown의 Redis 호출도 모킹
"""
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from io import BytesIO
from datetime import datetime

from fastapi.testclient import TestClient


# =============================================================================
# 헬퍼 함수 - 매번 새 BytesIO 생성 (closed file 문제 방지)
# =============================================================================
def make_image_file():
    """이미지 BytesIO를 새로 생성"""
    return BytesIO(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)


def make_pdf_file():
    """PDF BytesIO를 새로 생성"""
    return BytesIO(b'%PDF-1.4' + b'\x00' * 100)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def client():
    """
    테스트 클라이언트

    핵심: startup/shutdown 이벤트에서 호출되는 Redis 함수도 모킹해야
    TestClient context manager가 정상 동작함
    """
    with patch('app.main.get_redis', new_callable=AsyncMock, return_value=None), \
         patch('app.main.close_redis', new_callable=AsyncMock), \
         patch('app.main.health_check_redis', new_callable=AsyncMock,
               return_value={"status": "mocked", "version": "test"}):
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client


# =============================================================================
# 테스트: 비동기 작업 생성
# =============================================================================
class TestAsyncPredictSubmission:

    def test_returns_202_with_task_id(self, client):
        """
        POST /predict → 202 Accepted + task_id 반환
        """
        with patch('app.main.get_cached_result', new_callable=AsyncMock, return_value=None), \
             patch('app.main.set_task_status', new_callable=AsyncMock), \
             patch('app.main.is_db_available', return_value=True), \
             patch('app.main.extract_building_ledger', return_value={'address': '테스트'}), \
             patch('app.main.extract_real_estate_data', return_value={'address': '테스트'}), \
             patch('app.main.validate_document_match',
                   return_value=(True, "OK", {'confidence': 0.95, 'errors': [], 'match_scores': {}})), \
             patch('app.main.predict_risk_with_ocr', return_value={
                 "meta": {"code": 200, "message": "완료", "timestamp": datetime.now().isoformat()},
                 "data": {"risk_score": 35.0, "risk_level": "SAFE"}
             }), \
             patch('app.main.set_cached_result', new_callable=AsyncMock):

            response = client.post(
                "/predict",
                data={"deposit": 30000, "address": "인천광역시 부평구 삼산동 167-15"},
                files=[
                    ("ledger_files", ("test.png", make_image_file(), "image/png")),
                    ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf")),
                ]
            )

        assert response.status_code == 202

        data = response.json()
        assert data["meta"]["code"] == 202
        assert "task_id" in data["data"]
        assert data["data"]["status"] == "PENDING"
        assert "poll_url" in data["data"]

    def test_missing_files_returns_400(self, client):
        """
        파일 미첨부 → 400 Bad Request
        """
        response = client.post(
            "/predict",
            data={"deposit": 30000, "address": "인천광역시 부평구 삼산동 167-15"},
        )

        assert response.status_code == 400
        data = response.json()
        assert any("ledger_files" in e["field"] for e in data["errors"])
        assert any("registry_files" in e["field"] for e in data["errors"])

    def test_too_many_files_returns_400(self, client):
        """
        파일 개수 초과 → 400 Bad Request
        """
        response = client.post(
            "/predict",
            data={"deposit": 30000, "address": "인천광역시 부평구 삼산동 167-15"},
            files=[
                *[("ledger_files", (f"test{i}.png", make_image_file(), "image/png")) for i in range(6)],
                ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf")),
            ]
        )

        assert response.status_code == 400
        data = response.json()
        assert any("최대 5개" in e["message"] for e in data["errors"])


# =============================================================================
# 테스트: 캐시 히트
# =============================================================================
class TestCacheHit:

    def test_cached_result_returns_200(self, client):
        """
        동일 요청의 캐시가 있으면 → 즉시 200 OK 반환
        """
        cached_result = {
            "meta": {"code": 200, "message": "캐시된 결과", "cached": True},
            "data": {"risk_score": 41.0, "risk_level": "SAFE"}
        }

        with patch('app.main.get_cached_result', new_callable=AsyncMock,
                    return_value=cached_result), \
             patch('app.main.generate_cache_key', return_value="predict:test123"), \
             patch('app.main.generate_file_hash', return_value="abc123"):

            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "인천광역시 부평구 삼산동 167-15"
                },
                files=[
                    ("ledger_files", ("test.png", make_image_file(), "image/png")),
                    ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf")),
                ]
            )

        assert response.status_code == 200
        data = response.json()
        assert data["meta"].get("cached") is True

    def test_skip_cache_forces_new_analysis(self, client):
        """
        skip_cache=True → 캐시 무시, 새로 분석 (202)
        """
        cached_result = {
            "meta": {"code": 200, "message": "캐시된 결과"},
            "data": {"risk_score": 41.0}
        }

        with patch('app.main.get_cached_result', new_callable=AsyncMock,
                    return_value=cached_result), \
             patch('app.main.set_task_status', new_callable=AsyncMock), \
             patch('app.main.generate_cache_key', return_value="predict:test123"), \
             patch('app.main.generate_file_hash', return_value="abc123"):

            response = client.post(
                "/predict",
                data={
                    "deposit": 30000,
                    "address": "인천광역시 부평구 삼산동 167-15",
                    "skip_cache": "true",
                },
                files=[
                    ("ledger_files", ("test.png", make_image_file(), "image/png")),
                    ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf")),
                ]
            )

        # skip_cache=True이므로 캐시가 있어도 202 반환
        assert response.status_code == 202


# =============================================================================
# 테스트: 작업 상태 폴링
# =============================================================================
class TestTaskPolling:

    def test_completed_task_returns_result(self, client):
        """
        완료된 작업 조회 → 200 + result
        """
        task_data = {
            "task_id": "test-123",
            "status": "COMPLETED",
            "progress": 100,
            "result": {"risk_score": 35.0, "risk_level": "SAFE"}
        }

        with patch('app.main.get_task_status', new_callable=AsyncMock,
                    return_value=task_data):
            response = client.get("/predict/test-123")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "COMPLETED"
        assert data["data"]["result"]["risk_score"] == 35.0

    def test_processing_task_returns_progress(self, client):
        """
        진행 중인 작업 → 200 + progress
        """
        task_data = {
            "task_id": "test-456",
            "status": "PROCESSING",
            "progress": 60,
        }

        with patch('app.main.get_task_status', new_callable=AsyncMock,
                    return_value=task_data):
            response = client.get("/predict/test-456")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "PROCESSING"
        assert data["data"]["progress"] == 60

    def test_failed_task_returns_error(self, client):
        """
        실패한 작업 → 200 + error
        """
        task_data = {
            "task_id": "test-789",
            "status": "FAILED",
            "progress": 25,
            "error": "건축물대장 OCR 처리 실패"
        }

        with patch('app.main.get_task_status', new_callable=AsyncMock,
                    return_value=task_data):
            response = client.get("/predict/test-789")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "FAILED"
        assert "OCR" in data["data"]["error"]

    def test_unknown_task_returns_404(self, client):
        """
        존재하지 않는 작업 → 404
        """
        with patch('app.main.get_task_status', new_callable=AsyncMock,
                    return_value=None):
            response = client.get("/predict/nonexistent-id")

        assert response.status_code == 404


# =============================================================================
# 테스트: DB 연결 실패
# =============================================================================
class TestDBConnectionFailure:

    def test_db_failure_during_background_task(self, client):
        """
        백그라운드 작업 중 DB 실패 → 작업은 202로 생성됨
        (실제 FAILED 상태는 백그라운드에서 비동기 처리)
        """
        with patch('app.main.get_cached_result', new_callable=AsyncMock, return_value=None), \
             patch('app.main.set_task_status', new_callable=AsyncMock) as mock_set_status, \
             patch('app.main.is_db_available', return_value=False), \
             patch('app.main.reset_db_availability'), \
             patch('app.main.check_db_connection', return_value=False), \
             patch('app.main.generate_cache_key', return_value="predict:dbfail"), \
             patch('app.main.generate_file_hash', return_value="hash123"):

            response = client.post(
                "/predict",
                data={"deposit": 30000, "address": "테스트 주소"},
                files=[
                    ("ledger_files", ("test.png", make_image_file(), "image/png")),
                    ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf")),
                ]
            )

        # 요청 자체는 202 (작업 생성은 됨, 실패는 백그라운드에서 처리)
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data["data"]


# =============================================================================
# 테스트: 캐시 무효화
# =============================================================================
class TestCacheInvalidation:

    def test_delete_existing_cache(self, client):
        """
        존재하는 캐시 삭제 → 200
        """
        with patch('app.main.invalidate_cache', new_callable=AsyncMock,
                    return_value=True):
            response = client.delete("/predict/cache/predict:abc123")

        assert response.status_code == 200

    def test_delete_nonexistent_cache(self, client):
        """
        존재하지 않는 캐시 삭제 → 404
        """
        with patch('app.main.invalidate_cache', new_callable=AsyncMock,
                    return_value=False):
            response = client.delete("/predict/cache/predict:notfound")

        assert response.status_code == 404


# =============================================================================
# 테스트: 헬스체크
# =============================================================================
class TestHealthCheck:

    def test_health_check_includes_redis_status(self, client):
        """
        GET / → Redis 상태 포함
        """
        with patch('app.main.health_check_redis', new_callable=AsyncMock,
                    return_value={"status": "connected", "version": "7.0"}):
            response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "2.0"
        assert "redis" in data


# =============================================================================
# 테스트: 파일 타입 검증
# =============================================================================
class TestFileValidation:

    def test_invalid_image_type_returns_400(self, client):
        """
        잘못된 이미지 타입 → 400
        """
        response = client.post(
            "/predict",
            data={"deposit": 30000, "address": "인천광역시 부평구 삼산동 167-15"},
            files=[
                ("ledger_files", ("test.txt", BytesIO(b"not an image"), "text/plain")),
                ("registry_files", ("test.pdf", make_pdf_file(), "application/pdf")),
            ]
        )

        assert response.status_code == 400
        data = response.json()
        assert any("ledger_files" in e["field"] for e in data["errors"])

    def test_invalid_pdf_type_returns_400(self, client):
        """
        잘못된 PDF 타입 → 400
        """
        response = client.post(
            "/predict",
            data={"deposit": 30000, "address": "인천광역시 부평구 삼산동 167-15"},
            files=[
                ("ledger_files", ("test.png", make_image_file(), "image/png")),
                ("registry_files", ("test.txt", BytesIO(b"not a pdf"), "text/plain")),
            ]
        )

        assert response.status_code == 400
        data = response.json()
        assert any("registry_files" in e["field"] for e in data["errors"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])