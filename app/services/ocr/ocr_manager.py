from app.services.ocr.registry_parser import extract_real_estate_data  # 등기부등본
from app.services.ocr.ledger_parser import extract_building_ledger # 건축물대장

def parse_document(file_path, doc_type="unknown"):
    """
    doc_type: 'registry'(등기부) 또는 'ledger'(건축물대장)
    """
    if doc_type == "registry":
        return extract_real_estate_data(file_path)
    elif doc_type == "ledger":
        return extract_building_ledger(file_path)
    else:
        # 타입을 모르면? Gemini에게 "이거 무슨 문서야?"라고 먼저 묻는 로직을 추가하거나
        # 사용자에게 선택하게 하면 됩니다.
        return {"error": "문서 타입을 지정해주세요."}