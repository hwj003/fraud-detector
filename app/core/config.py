import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# [1] .env 파일 로드
# (프로젝트 루트 폴더에서 .env 파일을 찾음)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# [2] 현재 환경 읽기 (기본값: 'local')
APP_ENV = os.getenv("APP_ENV", "local")

# [3] DB_URL 및 SQLAlchemy 엔진 설정
DB_URL = ""
engine = None

print(f"--- [DB Config] 현재 환경: {APP_ENV} ---")

if APP_ENV == "prod":
    # --- 운영 환경 (MySQL) ---
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("[prod] DB 접속 정보(USER, PASS, HOST, PORT, NAME)가 없습니다.")

    DB_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"운영 DB(MySQL)에 연결합니다: {DB_HOST}")

else:
    # --- 로컬 환경 (SQLite) ---
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "local_dev.sqlite")

    # [중요] SQLite 경로는 프로젝트 루트 기준이어야 함
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DB_PATH = os.path.join(project_root, SQLITE_DB_PATH)

    DB_URL = f"sqlite:///{DB_PATH}"
    print(f"로컬 DB(SQLite)에 연결합니다: {DB_PATH}")

# [4] 최종 엔진 생성
try:
    engine = create_engine(DB_URL)
except Exception as e:
    print(f"DB 엔진 생성 실패: {e}")
    raise