import uvicorn
import os

if __name__ == "__main__":
    # 스크립트 실행 위치가 프로젝트 루트인지 확인 (권장)
    if not os.path.exists('app/main.py'):
        print("경고: 이 스크립트는 'fraud_detector' 루트 디렉토리에서 실행해야 합니다.")
        print("Running from current directory...")

    uvicorn.run(
        "app.main:app", # "모듈.파일:FastAPI객체"
        host="127.0.0.1",
        port=8000,
        reload=True
    )