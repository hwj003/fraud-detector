from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """환경 설정 - .env 파일에서 로드"""
    APP_ENV: str = "local"

    # MySQL 설정
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str
    DB_NAME: str

    # 연결 타임아웃 설정 (초)
    DB_CONNECT_TIMEOUT: int = 5
    DB_READ_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤"""
    return Settings()