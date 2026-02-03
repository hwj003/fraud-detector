from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import shutil
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from app.services.ocr.ledger_parser import extract_building_ledger
from app.services.ocr.registry_parser import extract_real_estate_data
from app.services.predict_service import predict_risk_with_ocr  # 새로 만들 함수
from app.router import stats
from app.core import (
    get_settings,
    check_db_connection,
    is_db_available,
    reset_db_availability
)
from app.core.exceptions import DatabaseConnectionError, ServiceUnavailableError
from app.services.document_validator import (
    validate_document_match
)

app = FastAPI()

settings = get_settings()
origins = ["*"] if settings.APP_ENV == "local" else [
    "https://your-frontend-domain.com",
    "https://app.your-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # 필요시
    allow_methods=["GET", "POST"],  # 필요한 것만
    allow_headers=["*"],
)

# 파일 업로드 제약사항
MAX_FILE_SIZE_MB = 20
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_PDF_TYPES = {"application/pdf"}


def validate_file_size(file: UploadFile, max_size_mb: int = MAX_FILE_SIZE_MB):
    """파일 크기 검증"""
    file.file.seek(0, 2)  # 파일 끝으로 이동
    file_size = file.file.tell()  # 현재 위치 = 파일 크기
    file.file.seek(0)  # 다시 처음으로

    if file_size > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기는 {max_size_mb}MB 이하여야 합니다 (현재: {file_size / 1024 / 1024:.1f}MB)"
        )


def validate_file_type(file: UploadFile, allowed_types: set):
    """파일 타입 검증"""
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"지원하지 않는 파일 형식입니다. 허용 형식: {', '.join(allowed_types)}"
        )


def create_error_response(
        status_code: int,
        message: str,
        errors: list,
        suggestions: list = None
) -> JSONResponse:
    """에러 응답 생성 헬퍼 함수"""
    content = {
        "meta": {
            "code": status_code,
            "message": message,
            "timestamp": datetime.now().isoformat()
        },
        "errors": errors
    }

    if suggestions:
        content["suggestions"] = suggestions

    return JSONResponse(status_code=status_code, content=content)


# ============================================================
# DB 연결 체크 의존성 (재사용 가능)
# ============================================================
async def verify_db_connection():
    """
    FastAPI 의존성: DB 연결 상태 확인

    OCR 분석 전에 DB가 정상적으로 연결되어 있는지 확인합니다.
    연결 실패 시 503 Service Unavailable 반환

    사용법:
        @app.post("/predict")
        async def predict_risk(..., _=Depends(verify_db_connection)):
            ...
    """
    # 캐시된 상태로 빠른 체크
    if is_db_available():
        return True

    # 캐시가 없거나 이전에 실패했다면 재확인
    reset_db_availability()

    if not check_db_connection():
        raise HTTPException(
            status_code=503,
            detail={
                "meta": {
                    "code": 503,
                    "message": "서비스를 일시적으로 사용할 수 없습니다",
                    "timestamp": datetime.now().isoformat()
                },
                "errors": [{
                    "field": "database",
                    "message": "데이터베이스 연결을 확인할 수 없습니다. 잠시 후 다시 시도해주세요."
                }],
                "suggestions": [
                    "네트워크 연결 상태를 확인해주세요",
                    "잠시 후 다시 시도해주세요",
                    "문제가 지속되면 관리자에게 문의해주세요"
                ]
            }
        )

    return True

@app.get("/", summary="서비스 상태 확인")
def health_check():
    return {
        "status": "Healthy",
        "service": "Fraud Detector AI",
        "version": "1.0"
    }

# 라우터 등록
app.include_router(stats.router)

@app.post("/predict",
          summary="파일 업로드 기반 정밀 분석",
          description="""
          건축물대장(이미지)과 등기부등본(PDF)을 업로드하여 정밀 위험도 분석을 수행합니다.
          
          **파일 제약사항:**
          - 건축물대장: PNG, JPG, JPEG (최대 10MB, 최대 5개)
          - 등기부등본: PDF (최대 20MB, 최대 3개)
          """)
async def predict_risk(
        deposit: int = Form(..., description="보증금 (만원)", ge=0, le=1000000),
        address: str = Form(..., description="주소 (시세 조회용)", min_length=5, max_length=200),
        ledger_files: List[UploadFile] = File(default=None, description="건축물대장 이미지 (PNG/JPG)"),
        registry_files: List[UploadFile] = File(default=None, description="등기부등본 파일 (PDF)")
):
    """
    개선된 파일 업로드 기반 위험도 분석

    **응답 예시:**
    ```json
    {
      "meta": {
        "code": 200,
        "message": "전세사기 위험도 분석 완료",
        "timestamp": "2026-02-02T14:30:45"
      },
      "data": {
        "address": "인천광역시 부평구 삼산동 167-15",
        "deposit": 35000000,
        "market_price": 61000000,
        "price_source": "DB_Trade",
        "risk_score": 41.0,
        "risk_level": "SAFE",
        "major_risk_factors": [...],
        "hug_result": {...},
        "details": {...},
        "recommendations": [...]
      }
    }
    ```
    """

    # === 입력 검증 ===
    errors = []

    # 건축물대장 필수 체크
    if not ledger_files or len(ledger_files) == 0:
        errors.append({
            "field": "ledger_files",
            "message": "건축물대장 파일은 필수입니다. 최소 1개 이상의 이미지를 업로드해주세요."
        })

    # 등기부등본 필수 체크
    if not registry_files or len(registry_files) == 0:
        errors.append({
            "field": "registry_files",
            "message": "등기부등본 파일은 필수입니다. 최소 1개 이상의 PDF를 업로드해주세요."
        })

    # 건축물대장 검증
    if ledger_files and len(ledger_files) > 0:
        if len(ledger_files) > 5:
            errors.append({
                "field": "ledger_files",
                "message": "건축물대장은 최대 5개까지 업로드 가능합니다"
            })

        for idx, file in enumerate(ledger_files):
            try:
                validate_file_type(file, ALLOWED_IMAGE_TYPES)
                validate_file_size(file, max_size_mb=10)
            except HTTPException as e:
                errors.append({
                    "field": f"ledger_files[{idx}]",
                    "message": f"{file.filename}: {e.detail}"
                })

    # 등기부등본 검증
    if registry_files and len(registry_files) > 0:
        if len(registry_files) > 3:
            errors.append({
                "field": "registry_files",
                "message": "등기부등본은 최대 3개까지 업로드 가능합니다"
            })

        for idx, file in enumerate(registry_files):
            try:
                validate_file_type(file, ALLOWED_PDF_TYPES)
                validate_file_size(file, max_size_mb=20)
            except HTTPException as e:
                errors.append({
                    "field": f"registry_files[{idx}]",
                    "message": f"{file.filename}: {e.detail}"
                })

    # 에러가 있으면 400 반환
    if errors:
        return create_error_response(400, "입력 데이터 검증 실패", errors)

    ocr_results = {
        'ledger': {},
        'registry': {}
    }

    # 1. 임시 파일 저장 경로 생성
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    ledger_paths = []
    registry_paths = []
    try:
        # OCR 처리 직전 DB 연결 재확인
        print("[API] DB 연결 상태 확인 중...", flush=True)
        if not is_db_available():
            # 캐시 리셋 후 재확인
            reset_db_availability()
            if not check_db_connection():
                return create_error_response(
                    503,
                    "서비스를 일시적으로 사용할 수 없습니다",
                    [{
                        "field": "database",
                        "message": "데이터베이스 연결을 확인할 수 없습니다"
                    }],
                    suggestions=["잠시 후 다시 시도해주세요"]
                )
        print("[API] DB 연결 확인 완료", flush=True)
        # 1. 건축물대장 저장 및 분석
        if ledger_files:
            for file in ledger_files:
                file_path = os.path.join(temp_dir, f"ledger_{datetime.now().timestamp()}_{file.filename}")
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                ledger_paths.append(file_path)

            try:
                ocr_results['ledger'] = extract_building_ledger(ledger_paths)
            except Exception as e:
                return create_error_response(
                    422,
                    "파일 분석 실패",
                    [{
                        "field": "ledger_files",
                        "message": f"건축물대장 OCR 처리 실패: {str(e)}. 선명한 이미지를 업로드해주세요"
                    }]
                )

        # 2. 등기부등본 저장 및 분석
        if registry_files:
            for file in registry_files:
                file_path = os.path.join(temp_dir, f"registry_{datetime.now().timestamp()}_{file.filename}")
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                registry_paths.append(file_path)

            try:
                ocr_results['registry'] = extract_real_estate_data(registry_paths)
            except Exception as e:
                return create_error_response(
                    422,
                    "파일 분석 실패",
                    [{
                        "field": "registry_files",
                        "message": f"등기부등본 OCR 처리 실패: {str(e)}. 선명한 PDF를 업로드해주세요"
                    }]
                )

        # ================================================================
        # 3. 문서 매칭 검증
        # ================================================================
        if ledger_files and registry_files:
            ledger_data = ocr_results.get('ledger', {})
            registry_data = ocr_results.get('registry', {})

            # 둘 다 유효한 데이터가 있을 때만 검증
            if ledger_data and registry_data:
                # 간편 함수 사용
                is_valid, message, details = validate_document_match(
                    ledger_data,
                    registry_data
                )

                if not is_valid:
                    # 문서 불일치 에러 응답
                    return create_error_response(
                        status_code=422,
                        message="문서 불일치 오류",
                        errors=[{
                            "field": "documents",
                            "message": message,
                            "details": {
                                "confidence": f"{details['confidence']:.1%}",
                                "issues": details['errors'],
                                "match_scores": {
                                    k: f"{v:.1%}"
                                    for k, v in details['match_scores'].items()
                                }
                            }
                        }],
                        suggestions=[
                            "건축물대장과 등기부등본이 같은 주소의 문서인지 확인해주세요",
                            "호수(동/호)가 정확히 일치하는지 확인해주세요",
                            "최신 문서를 사용하고 있는지 확인해주세요"
                        ]
                    )

                # 경고가 있으면 로그 기록 (검증은 통과)
                if details.get('warnings'):
                    print(f"[문서검증 경고] {details['warnings']}", flush=True)

        # 4. 예측 실행
        print(f"[API] 예측 시작 - 주소: {address}, 보증금: {deposit}만원", flush=True)
        result = predict_risk_with_ocr(address, deposit, ocr_results)
        print("[API] 예측 완료", flush=True)

        return result
    except DatabaseConnectionError as e:
        # DB 연결 오류 - 명확한 500 에러 반환
        print(f"[에러] 데이터베이스 연결 실패: {e}", flush=True)

        return create_error_response(
            500,
            "서버 오류가 발생했습니다",
            [{
                "field": "server",
                "message": f"분석 실패: {str(e)}"
            }]
        )

    except ServiceUnavailableError as e:
        # 서비스 사용 불가 오류
        print(f"[에러] 서비스 사용 불가: {e}", flush=True)

        return create_error_response(
            503,
            "서비스를 일시적으로 사용할 수 없습니다",
            [{
                "field": "service",
                "message": str(e)
            }]
        )
    except Exception as e:
        print(f"[에러] 예측 중 오류: {e}", flush=True)
        import traceback
        traceback.print_exc()

        # 연결 관련 오류 메시지 확인
        error_msg = str(e)

        if "Can't connect" in error_msg or "Connection refused" in error_msg:
            return create_error_response(
                500,
                "서버 오류가 발생했습니다",
                [{
                    "field": "server",
                    "message": "분석 실패: Database connection failed"
                }]
            )

        return create_error_response(
            500,
            "서버 오류가 발생했습니다",
            [{
                "field": "server",
                "message": f"분석 실패: {str(e)}"
            }]
        )

    finally:
        # 임시 파일 정리
        for path in ledger_paths + registry_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[Warning] 임시 파일 삭제 실패: {path} - {e}")


if __name__ == "__main__":
    import uvicorn
    # 로컬 개발용 실행 커맨드: python app/main.py
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)