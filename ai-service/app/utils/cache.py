# cache.py - UPDATED VERSION
import redis.asyncio as redis
import json
import logging
from typing import Optional, Any, Dict
from app.utils.config import settings
from contextlib import asynccontextmanager
import asyncio
logger = logging.getLogger(__name__)

class RedisConnectionPool:
    """Production Redis connection pool with health checks"""
    
    _instance = None
    _client = None
    _is_connected = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self, max_retries: int = 3):
        """Initialize Redis with connection pool and retry logic"""
        if self._is_connected and self._client:
            return self._client
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Production-grade connection pool
                self._client = await redis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    health_check_interval=30,
                    retry_on_timeout=True,
                    max_connections=50,
                    socket_timeout=10
                )
                
                # Test connection with timeout
                await asyncio.wait_for(self._client.ping(), timeout=5)
                self._is_connected = True
                
                logger.info(f"✅ Redis connected successfully (attempt {retry_count + 1}/{max_retries})")
                return self._client
                
            except (redis.ConnectionError, asyncio.TimeoutError) as e:
                retry_count += 1
                logger.warning(f"Redis connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                continue
            except Exception as e:
                logger.error(f"Redis initialization error: {e}")
                self._client = None
                self._is_connected = False
                raise
        
        logger.error(f"Failed to connect to Redis after {max_retries} attempts")
        self._client = None
        self._is_connected = False
        return None
    
    async def get_client(self):
        """Get Redis client, initialize if needed"""
        if not self._client or not self._is_connected:
            await self.initialize()
        return self._client
    
    async def health_check(self):
        """Check if Redis is still connected"""
        try:
            client = await self.get_client()
            if client:
                await client.ping()
                return True
            return False
        except:
            self._is_connected = False
            return False
    
    async def close(self):
        """Close Redis connection properly"""
        if self._client:
            await self._client.close()
            self._client = None
            self._is_connected = False
            logger.info("Redis connection closed")

# Singleton instance
redis_pool = RedisConnectionPool()

async def get_redis_client() -> Optional[redis.Redis]:
    """Dependency injection for Redis client"""
    try:
        return await redis_pool.get_client()
    except Exception as e:
        logger.error(f"Failed to get Redis client: {e}")
        return None

async def safe_redis_operation(func, *args, **kwargs):
    """Safely execute Redis operations with fallback"""
    try:
        client = await get_redis_client()
        if client:
            return await func(client, *args, **kwargs)
    except Exception as e:
        logger.error(f"Redis operation failed: {e}")
    return None

# Global redis_client with fallback behavior
async def redis_client():
    """Global Redis client with lazy initialization"""
    return await get_redis_client()

async def init_redis():
    """Initialize Redis on application startup"""
    try:
        import asyncio
        client = await redis_pool.initialize()
        if client:
            logger.info("Redis initialized successfully")
            return True
        logger.error("Redis initialization failed")
        return False
    except Exception as e:
        logger.error(f"Redis initialization error: {e}")
        return False

async def close_redis():
    """Close Redis on application shutdown"""
    await redis_pool.close()