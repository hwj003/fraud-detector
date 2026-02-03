import os
import json
import mimetypes
import pprint
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


def extract_real_estate_data(file_paths: list):
    """
    등기부등본(Registry) 이미지를 분석하여 JSON으로 추출합니다.
    여러 페이지를 리스트로 받아 통합 분석하며, 상세 디버깅 로그를 출력합니다.
    """

    # [DEBUG] 시작 로그
    print("\n" + "=" * 60)
    print(f"🕵️‍♂️ [DEBUG] 등기부등본 OCR 분석 시작")
    print(f"📂 [DEBUG] 입력된 파일 개수: {len(file_paths)}개")

    if not file_paths:
        print("❌ [DEBUG] 파일 경로 리스트가 비어있습니다.")
        return {}

    try:
        client = genai.Client(api_key=API_KEY)

        # 1. Gemini에게 보낼 컨텐츠 리스트 생성
        content_parts = []

        # 2. 모든 이미지 파일을 순회하며 Part 생성
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

            # 리스트에 추가
            content_parts.append(
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            )

        if not content_parts:
            print("❌ [DEBUG] 유효한 파일 내용이 없어 중단합니다.")
            return {}

        # 3. 프롬프트 설정
        prompt = """
                당신은 깐깐한 부동산 권리 분석가입니다.
                제공된 등기부등본 이미지를 **[논리적 3단계]**로 분석하여 결과를 도출하세요.

                **[1단계: 말소(삭제)된 권리 식별]**
                문서 전체에서 '등기목적' 열을 확인하여 **말소**, **해지**, **해제**라는 단어가 포함된 행을 모두 찾으세요.
                그리고 그 행이 **"몇 번 순위번호"를 지우는지** 파악하세요.
                (예: "1번근저당권설정등기말소" -> 순위번호 1번은 삭제됨. 따라서 1-1번 같은 부기등기도 함께 삭제됨.)
                (예: "2번압류등기말소" -> 순위번호 2번은 삭제됨.)

                **[2단계: 유효한 권리 필터링]**
                이제 '갑구'(소유권)와 '을구'(소유권 이외)의 모든 권리를 확인하되, 
                **1단계에서 파악된 '삭제된 순위번호'에 해당하는 권리는 과감히 버리세요.**
                빨간 줄(삭선)이 그어져 있는 경우도 당연히 버리세요.

                **[3단계: 최종 데이터 추출]**
                2단계에서 살아남은(유효한) 권리만으로 JSON을 구성하세요.

                --------------------------------------------------

                **[추출 항목 및 JSON 구조]**
                1. **basic_info** (기본 및 소유자 정보):
                   - `address`: 표제부의 소재지 (지번 주소 우선).
                   - `owner`: 갑구의 **최종 유효** 소유자 이름.
                   - `ownership_date`: 최종 소유자의 등기접수일 (YYYY-MM-DD).

                2. **risk_factors** (소유권 침해 및 위험 등기):
                   - `trust_content`: 갑구 '신탁' 등기 여부 (없으면 "없음").
                   - `injunction_content`: 갑구/을구 '압류, 가압류, 가처분, 경매' 등기 내용. **(단, 말소된 건 제외)**
                   - `lease_order_content`: 을구 '임차권등기명령' 내용. **(단, 말소된 건 제외)**

                3. **debts** (채무 및 선순위 권리 목록 - 을구):
                   - **반드시 말소되지 않은 유효한 근저당/전세권만 추출하세요.**
                   - `type`: "근저당" 또는 "전세권"
                   - `amount`: 채권최고액 또는 전세금 (숫자만, 예: 150000000).
                   - `creditor`: 권리자 이름.
                   - `date`: 등기 접수일 (YYYY-MM-DD).

                **[출력 예시 - 빚이 없는 경우]**
                {
                  "basic_info": { ... },
                  "risk_factors": { "trust_content": "없음", "injunction_content": "없음", "lease_order_content": "없음" },
                  "debts": [] 
                }
                """

        # [DEBUG] API 호출 직전 로그
        print(f"🚀 [DEBUG] Gemini API 요청 전송... (총 {len(content_parts)}개 파트)")

        # [수정됨] content_parts 리스트 전체를 보내도록 수정 (기존 코드 버그 수정)
        # 텍스트 프롬프트도 리스트에 추가
        content_parts.append(prompt)

        # API 호출
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content_parts,  # 수정된 부분: 리스트 전체 전달
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )

        # [DEBUG] 응답 결과 확인
        print("📥 [DEBUG] Gemini 응답 수신 완료")
        print(response, flush=True)
        if not response.text:
            print("❌ [DEBUG] 응답 텍스트가 비어있습니다.")
            return {}

        # 결과 파싱
        try:
            parsed_json = json.loads(response.text)
            print("✅ [DEBUG] JSON 파싱 성공:")
            pprint.pprint(parsed_json)  # 예쁘게 출력
            print("=" * 60 + "\n")
            return parsed_json
        except json.JSONDecodeError as je:
            print(f"❌ [DEBUG] JSON 파싱 실패: {je}")
            print(f"   [Raw Text]: {response.text}")
            return {}

    except Exception as e:
        print(f"❌ [DEBUG] OCR 처리 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- 테스트 실행 블록 ---
if __name__ == "__main__":
    # 테스트용 파일 경로 설정 (자신의 환경에 맞게 수정 필요)
    # 예: script_dir = os.path.dirname(os.path.abspath(__file__)) ...

    # 임시 테스트 경로 리스트
    test_files = [
        # r"C:\path\to\test_image_page1.jpg",
        # r"C:\path\to\test_image_page2.jpg"
    ]

    if test_files:
        extract_real_estate_data(test_files)
    else:
        print("[System] 테스트할 파일 경로를 코드 하단 'test_files' 리스트에 넣어주세요.")