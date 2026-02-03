import sys
import io
import os
import mimetypes
import json
import pprint
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 환경 변수 로드 및 인코딩 설정
load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
API_KEY = os.getenv("GEMINI_API_KEY")


def extract_building_ledger(file_paths: list):
    """
    건축물대장(Building Ledger)을 분석하여 핵심 정보를 JSON으로 추출합니다.
    여러 페이지(이미지/PDF)를 리스트로 받아 하나의 문서로 통합 분석하며, 상세 디버그 로그를 출력합니다.
    """

    # [DEBUG] 시작 로그
    print("\n" + "=" * 60)
    print(f"🏗️ [DEBUG] 건축물대장 OCR 분석 시작")
    print(f"📂 [DEBUG] 입력된 파일 개수: {len(file_paths)}개")

    # 1. 입력값 검증
    if not file_paths:
        print("❌ [DEBUG] 파일 경로 리스트가 비어있습니다.")
        return {"error": "파일 경로 리스트가 비어 있습니다."}

    try:
        client = genai.Client(api_key=API_KEY)

        # 2. 컨텐츠 파트 리스트 생성 (다중 이미지 취합)
        content_parts = []

        for idx, path in enumerate(file_paths):
            print(f"   📄 [DEBUG] 파일 처리 중 ({idx + 1}/{len(file_paths)}): {os.path.basename(path)}")

            if not os.path.exists(path):
                print(f"      ⚠️ [Warning] 파일이 존재하지 않습니다: {path}")
                continue

            with open(path, "rb") as f:
                file_bytes = f.read()
                print(f"      - 파일 크기: {len(file_bytes):,} bytes")

            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                ext = os.path.splitext(path)[1].lower()
                mime_type = 'application/pdf' if ext == '.pdf' else 'image/jpeg'

            print(f"      - MIME 타입: {mime_type}")

            # 리스트에 이미지/PDF 파트 추가
            content_parts.append(
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            )

        if not content_parts:
            print("❌ [DEBUG] 유효한 파일 내용이 없어 중단합니다.")
            return {"error": "유효한 파일이 없습니다."}

        # 3. 프롬프트 설정 (다중 페이지 안내 포함)
        prompt_text = """
                당신은 건축물대장 분석 AI입니다.
        제공된 이미지들은 **하나의 건축물대장 문서를 구성하는 여러 페이지**들입니다.
        페이지 순서대로 내용을 통합하여 전세 사기 위험도 분석에 필요한 핵심 정보를 추출하세요.

        **[분석 지침]**
        1. 문서는 표 형태로 되어 있습니다. 행과 열의 구조를 잘 파악하여 값을 추출하세요.
        2. 값이 없으면 null 또는 "없음"으로 표기하세요.
        3. **[중요] 위반건축물 여부**:
           - 문서의 1페이지 상단(제목 주변) 또는 '변동사항' 란에 '위반건축물' 표기가 있는지 확인하세요.
        4. **[중요] 사용승인일 추출 주의**:
           - 발급일자/열람일시를 가져오지 말고 표 안쪽의 '사용승인일'을 가져오세요.
        5. **[매우 중요] 주용도(main_usage) 추출 규칙**: ⭐
           - 반드시 **1페이지**의 **[전유부분]** 표에 있는 **'용도'** 란의 텍스트(예: '다세대주택')를 최우선으로 추출하세요.
           - 2페이지 등에 있는 **'공동주택(아파트) 가격'** 이라는 문구는 표의 제목일 뿐 건물의 용도가 아닙니다. **절대 이 제목을 main_usage로 가져오지 마세요.**
           - 만약 1페이지의 '용도'와 2페이지의 표 제목이 다르다면, 무조건 **1페이지 [전유부분]의 '용도'**를 정답으로 선택하세요.

                **[추출할 JSON 필드 정의]**
                1. **document_info**:
                   - `type`: 문서 종류 (예: "집합건축물대장(전유부, 갑)")
                   - `issue_date`: 발급일자 (YYYY-MM-DD) - 문서 우측 상단 등의 발급/열람 날짜
                   - `unique_number`: 상단의 '고유번호' 란에 적힌 숫자 (건물ID 아님! 예: 2823710200-3-03540011).

                2. **location**:
                   - `address`: 도로명 주소 (없으면 지번 주소).
                   - `detail_address`: 상세 주소 (호명칭 등, 예: "1101호").

                3. **building_status** (건물 현황):
                   - `main_usage`: '주용도'란에 적힌 내용 (예: "공동주택(아파트)", "제2종근린생활시설").
                   - `roof`: '지붕구조' (예: "철근콘크리트").
                   - `area`: '전유부분 면적' 숫자만 (예: 84.95).
                   - `usage_approval_date`: **'사용승인일'**란에 적힌 날짜 (YYYY-MM-DD).

                4. **safety_check** (안전 진단):
                   - `is_violator`: **위반건축물** 표기 여부 (true/false).
                   - `owner_name`: '소유자 현황'의 최종 소유자 이름.
                   - `ownership_date`: 소유권 변동일자 (YYYY-MM-DD).

                **[JSON 출력 예시]**
                {
                  "document_info": { 
                    "type": "집합건축물대장", 
                    "issue_date": "2024-01-01",
                    "unique_number": "41150-10100-20011-1234" 
                  },
                  "location": { "address": "서울시 강남구...", "detail_address": "101호" },
                  "building_status": { 
                    "main_usage": "다세대주택", 
                    "roof": "철근콘크리트", 
                    "area": 50.5,
                    "usage_approval_date": "1995-05-20"
                  },
                  "safety_check": { "is_violator": false, "owner_name": "홍길동", "ownership_date": "2020-05-05" }
                }
                """

        # 4. 프롬프트 텍스트 추가
        content_parts.append(prompt_text)

        # [DEBUG] API 호출 직전 로그
        print(f"🚀 [DEBUG] Gemini API 요청 전송... (총 {len(content_parts)}개 파트)")

        # 5. API 호출
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content_parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        print(response, flush=True)
        # [DEBUG] 응답 결과 확인
        print("📥 [DEBUG] Gemini 응답 수신 완료")

        if not response.text:
            print("❌ [DEBUG] 응답 텍스트가 비어있습니다.")
            return {"error": "API 응답이 비어있습니다."}

        # 6. 결과 파싱
        try:
            parsed_json = json.loads(response.text)
            print("✅ [DEBUG] JSON 파싱 성공:")
            pprint.pprint(parsed_json)  # 예쁘게 출력
            print("=" * 60 + "\n")
            return parsed_json

        except json.JSONDecodeError as je:
            print(f"❌ [DEBUG] JSON 파싱 실패: {je}")
            print(f"   [Raw Text]: {response.text}")
            return {"error": "JSON 파싱 실패"}

    except Exception as e:
        print(f"❌ [DEBUG] OCR 처리 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# --- 테스트 실행 ---
if __name__ == "__main__":

    # 테스트용 파일 경로 리스트
    # 실제 테스트할 때는 여기에 파일 경로들을 넣으세요.
    test_files = [
        # r"C:\Users\...\Desktop\ledger_sample_1.jpg",
        # r"C:\Users\...\Desktop\ledger_sample_2.jpg"
    ]

    if test_files:
        extract_building_ledger(test_files)
    else:
        print("[System] 테스트할 파일 경로를 코드 하단 'test_files' 리스트에 넣어주세요.")