import os
import sys
from google import genai
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ API 키를 찾을 수 없습니다. .env 파일을 확인해주세요.")
    sys.exit(1)


def list_all_models_safely():
    print(f"🔑 API Key 확인됨. 전체 모델 목록 조회 중...\n")

    try:
        # 클라이언트 초기화
        client = genai.Client(api_key=API_KEY)

        # 모델 리스트 가져오기
        pager = client.models.list()

        print(f"{'Model ID (이걸 복사해서 쓰세요)':<40} | {'Display Name'}")
        print("=" * 70)

        # 필터링 없이 일단 다 출력
        for model in pager:
            # 에러 방지를 위해 getattr 사용
            name = getattr(model, 'name', 'Unknown ID')
            display_name = getattr(model, 'display_name', '')

            # 모델 이름에 'gemini'가 들어간 것만 출력 (보기도 편하게)
            if 'gemini' in str(name).lower():
                # 'models/' 접두사가 있다면 보기 좋게 제거하고 출력할 수도 있지만,
                # 코드는 원본 ID를 아는게 중요하므로 그대로 출력
                print(f"{str(name):<40} | {str(display_name)}")

        print("=" * 70)
        print("\n💡 팁: 목록에 'gemini-1.5-flash'가 보이면,")
        print("   코드에서 model='gemini-1.5-flash' 라고 적으시면 됩니다.")
        print("   (앞에 'models/'가 붙어있다면 떼고 적으셔도 대부분 동작합니다.)")

    except Exception as e:
        print(f"\n❌ 모델 조회 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    list_all_models_safely()