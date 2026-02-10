"""
Redis 연결 및 캐시 관리

담당 기능:
- Redis 연결 관리 (싱글톤)
- 예측 결과 캐싱
- 작업 상태 관리 (비동기 작업 추적)
- 캐시 키 생성 유틸리티
"""
import json
import hashlib
import logging
from typing import Any, Optional, Dict
from datetime import datetime

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# =============================================================================
# Redis 설정 상수
# =============================================================================
CACHE_TTL_SECONDS = 3600          # 예측 결과 캐시: 1시간
TASK_TTL_SECONDS = 86400          # 작업 상태: 24시간
CACHE_KEY_PREFIX = "predict:"     # 캐시 키 접두사
TASK_KEY_PREFIX = "task:"         # 작업 상태 키 접두사

# =============================================================================
# 작업 상태 상수
# =============================================================================
class TaskStatus:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# =============================================================================
# Redis 연결 관리 (싱글톤)
# =============================================================================
_redis_client: Optional[Redis] = None


async def get_redis() -> Optional[Redis]:
    """
    Redis 비동기 클라이언트 (싱글톤)

    Returns:
        Redis 클라이언트 또는 None (연결 실패 시)
    """
    global _redis_client

    if _redis_client is not None:
        try:
            await _redis_client.ping()
            return _redis_client
        except (RedisConnectionError, RedisError):
            logger.warning("Redis 연결 끊김, 재연결 시도...")
            _redis_client = None

    try:
        settings = get_settings()
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')

        _redis_client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5,
            retry_on_timeout=True,
            max_connections=20,
        )

        await _redis_client.ping()
        logger.info(f"✅ Redis 연결 성공: {redis_url}")
        return _redis_client

    except (RedisConnectionError, RedisError, OSError) as e:
        logger.warning(f"⚠️ Redis 연결 실패 (캐시 없이 진행): {e}")
        _redis_client = None
        return None


async def close_redis():
    """Redis 연결 종료 (앱 셧다운 시 호출)"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis 연결 종료")


# =============================================================================
# 캐시 키 생성
# =============================================================================
def generate_cache_key(address: str, deposit: int, file_hashes: Optional[list] = None) -> str:
    """
    예측 요청에 대한 고유 캐시 키 생성

    Args:
        address: 주소
        deposit: 보증금 (만원)
        file_hashes: 업로드 파일의 해시 목록 (선택)

    Returns:
        캐시 키 문자열 (예: "predict:a1b2c3d4e5f6...")
    """
    raw = f"{address.strip()}:{deposit}"
    if file_hashes:
        raw += ":" + ":".join(sorted(file_hashes))

    hash_val = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX}{hash_val}"


def generate_file_hash(file_content: bytes) -> str:
    """파일 내용 기반 해시 생성"""
    return hashlib.md5(file_content).hexdigest()[:12]


# =============================================================================
# 예측 결과 캐시 조회/저장
# =============================================================================
async def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    캐시된 예측 결과 조회

    Args:
        cache_key: 캐시 키

    Returns:
        캐시된 결과 딕셔너리 또는 None
    """
    redis_client = await get_redis()
    if not redis_client:
        return None

    try:
        cached = await redis_client.get(cache_key)
        if cached:
            logger.info(f"🎯 캐시 히트: {cache_key}")
            result = json.loads(cached)
            # 캐시 응답임을 표시
            if "meta" in result:
                result["meta"]["cached"] = True
                result["meta"]["cache_key"] = cache_key
            return result
        return None

    except (RedisError, json.JSONDecodeError) as e:
        logger.warning(f"캐시 조회 실패: {e}")
        return None


async def set_cached_result(
        cache_key: str,
        result: Dict[str, Any],
        ttl: int = CACHE_TTL_SECONDS
) -> bool:
    """
    예측 결과를 캐시에 저장

    Args:
        cache_key: 캐시 키
        result: 저장할 결과
        ttl: TTL (초)

    Returns:
        저장 성공 여부
    """
    redis_client = await get_redis()
    if not redis_client:
        return False

    try:
        # datetime 직렬화 처리
        serializable = _make_serializable(result)
        await redis_client.setex(cache_key, ttl, json.dumps(serializable, ensure_ascii=False))
        logger.info(f"💾 캐시 저장: {cache_key} (TTL: {ttl}초)")
        return True

    except (RedisError, TypeError) as e:
        logger.warning(f"캐시 저장 실패: {e}")
        return False


async def invalidate_cache(cache_key: str) -> bool:
    """캐시 무효화"""
    redis_client = await get_redis()
    if not redis_client:
        return False

    try:
        deleted = await redis_client.delete(cache_key)
        if deleted:
            logger.info(f"🗑️ 캐시 삭제: {cache_key}")
        return bool(deleted)
    except RedisError as e:
        logger.warning(f"캐시 삭제 실패: {e}")
        return False


# =============================================================================
# 작업 상태 관리 (비동기 작업 추적)
# =============================================================================
async def set_task_status(
        task_id: str,
        status: str,
        progress: int = 0,
        result: Optional[Dict] = None,
        error: Optional[str] = None
) -> bool:
    """
    비동기 작업 상태 저장

    Args:
        task_id: 작업 고유 ID
        status: 상태 (PENDING, PROCESSING, COMPLETED, FAILED)
        progress: 진행률 (0~100)
        result: 완료된 결과 (COMPLETED 시)
        error: 에러 메시지 (FAILED 시)
    """
    redis_client = await get_redis()
    if not redis_client:
        return False

    task_data = {
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    if result is not None:
        task_data["result"] = _make_serializable(result)
    if error is not None:
        task_data["error"] = error

    try:
        key = f"{TASK_KEY_PREFIX}{task_id}"
        await redis_client.setex(key, TASK_TTL_SECONDS, json.dumps(task_data, ensure_ascii=False))
        return True
    except RedisError as e:
        logger.warning(f"작업 상태 저장 실패: {e}")
        return False


async def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """비동기 작업 상태 조회"""
    redis_client = await get_redis()
    if not redis_client:
        return None

    try:
        key = f"{TASK_KEY_PREFIX}{task_id}"
        data = await redis_client.get(key)
        return json.loads(data) if data else None
    except (RedisError, json.JSONDecodeError) as e:
        logger.warning(f"작업 상태 조회 실패: {e}")
        return None


# =============================================================================
# 유틸리티
# =============================================================================
def _make_serializable(obj: Any) -> Any:
    """JSON 직렬화 가능하도록 변환"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    return obj


async def health_check_redis() -> Dict[str, Any]:
    """Redis 헬스체크"""
    redis_client = await get_redis()
    if not redis_client:
        return {"status": "disconnected", "message": "Redis 연결 불가"}

    try:
        info = await redis_client.info("server")
        return {
            "status": "connected",
            "version": info.get("redis_version", "unknown"),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
        }
    except RedisError as e:
        return {"status": "error", "message": str(e)}