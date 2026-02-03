"""
OCR 파싱 모듈

건축물대장과 등기부등본을 Gemini API로 분석하여
구조화된 JSON 데이터로 추출합니다.

사용법:
    from app.services.ocr import extract_building_ledger, extract_real_estate_data

    # 건축물대장 분석
    ledger_data = extract_building_ledger(['/path/to/ledger1.jpg', '/path/to/ledger2.jpg'])

    # 등기부등본 분석
    registry_data = extract_real_estate_data(['/path/to/registry.pdf'])

    # 문서 타입 자동 선택
    data = parse_document(['/path/to/file.jpg'], doc_type='ledger')
"""
from app.services.ocr.ledger_parser import extract_building_ledger
from app.services.ocr.registry_parser import extract_real_estate_data
from app.services.ocr.ocr_manager import parse_document

__all__ = [
    "extract_building_ledger",   # 건축물대장 OCR
    "extract_real_estate_data",  # 등기부등본 OCR
    "parse_document",            # 통합 파서
]