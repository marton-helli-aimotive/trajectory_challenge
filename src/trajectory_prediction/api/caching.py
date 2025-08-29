"""
Prediction caching system for trajectory prediction API.

This module provides:
- In-memory prediction caching with LRU eviction
- Redis-based distributed caching
- Cache key generation and invalidation
- Cache statistics and monitoring
- TTL-based expiration
"""

import asyncio
import hashlib
import json
import time
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from ..data.schemas import TrajectoryData

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """
    In-memory LRU cache implementation.
    
    Thread-safe cache with size limits and TTL support.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired():
                    await self._remove_key(key)
                    self.miss_count += 1
                    return None
                
                # Update access statistics
                entry.update_access()
                
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                
                self.hit_count += 1
                return entry.value
            else:
                self.miss_count += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            current_time = time.time()
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0
            
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # If key already exists, remove from access order
            if key in self._cache:
                self._access_order.remove(key)
            
            # Add to cache
            self._cache[key] = entry
            self._access_order.append(key)
            
            # Evict if necessary
            await self._evict_if_necessary()
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_key(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    async def _remove_key(self, key: str) -> None:
        """Remove key from cache (internal method)."""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
    
    async def _evict_if_necessary(self) -> None:
        """Evict entries if cache is over capacity."""
        while len(self._cache) > self.max_size:
            # Remove least recently used
            lru_key = self._access_order[0]
            await self._remove_key(lru_key)
            self.eviction_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        total_size = sum(entry.size_bytes for entry in self._cache.values())
        
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "eviction_count": self.eviction_count,
            "total_size_bytes": total_size,
            "average_entry_size": total_size / len(self._cache) if len(self._cache) > 0 else 0
        }


class RedisCache:
    """
    Redis-based distributed cache implementation.
    
    Provides distributed caching with TTL support and atomic operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.config = config
        self.default_ttl = config.get("default_ttl", 3600)  # 1 hour
        
        # Redis connection parameters
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.key_prefix = config.get("key_prefix", "trajectory_prediction:")
        
        self.redis_client = None
        
        # Statistics tracking
        self.hit_count = 0
        self.miss_count = 0
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False  # Keep binary for pickle
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)
            
            if data is not None:
                self.hit_count += 1
                return pickle.loads(data)
            else:
                self.miss_count += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            redis_key = self._make_key(key)
            data = pickle.dumps(value)
            
            ttl_seconds = int(ttl or self.default_ttl)
            
            await self.redis_client.setex(redis_key, ttl_seconds, data)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self.redis_client:
            await self.connect()
        
        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all keys with prefix."""
        if not self.redis_client:
            await self.connect()
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
                
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "redis_host": self.host,
            "redis_port": self.port
        }


class PredictionCache:
    """
    High-level prediction caching system.
    
    Supports both in-memory and Redis-based caching with intelligent fallback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Cache configuration
        self.cache_type = config.get("type", "memory")  # "memory", "redis", "hybrid"
        self.default_ttl = config.get("default_ttl", 300)  # 5 minutes
        
        # Initialize cache backends
        self.memory_cache = None
        self.redis_cache = None
        
        if self.cache_type in ["memory", "hybrid"]:
            memory_config = config.get("memory", {})
            max_size = memory_config.get("max_size", 1000)
            self.memory_cache = LRUCache(max_size, self.default_ttl)
            
        if self.cache_type in ["redis", "hybrid"]:
            redis_config = config.get("redis", {})
            try:
                self.redis_cache = RedisCache(redis_config)
            except ImportError:
                logger.warning("Redis not available, falling back to memory cache")
                if self.memory_cache is None:
                    self.memory_cache = LRUCache(1000, self.default_ttl)
        
        # Cache key generation
        self.key_components = config.get("key_components", [
            "trajectory_hash", "prediction_horizon", "model_name"
        ])
        
        logger.info(f"Prediction cache initialized with type: {self.cache_type}")
    
    def generate_cache_key(
        self,
        trajectory_data: TrajectoryData,
        prediction_horizon: float,
        model_name: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for prediction request."""
        
        key_parts = []
        
        # Trajectory hash
        if "trajectory_hash" in self.key_components:
            trajectory_hash = self._hash_trajectory(trajectory_data)
            key_parts.append(f"traj_{trajectory_hash}")
        
        # Prediction horizon
        if "prediction_horizon" in self.key_components:
            horizon_str = f"horizon_{prediction_horizon:.2f}"
            key_parts.append(horizon_str)
        
        # Model name
        if "model_name" in self.key_components and model_name:
            key_parts.append(f"model_{model_name}")
        
        # Additional context
        if additional_context:
            context_hash = self._hash_dict(additional_context)
            key_parts.append(f"ctx_{context_hash}")
        
        cache_key = "|".join(key_parts)
        
        # Ensure key is not too long (Redis key limit is 512MB, but keep reasonable)
        if len(cache_key) > 250:
            cache_key = hashlib.sha256(cache_key.encode()).hexdigest()
        
        return cache_key
    
    def _hash_trajectory(self, trajectory_data: TrajectoryData) -> str:
        """Create hash of trajectory data for cache key."""
        
        # Create deterministic representation
        trajectory_repr = {
            "vehicle_id": trajectory_data.vehicle_id,
            "positions": trajectory_data.positions,
            "time_steps": trajectory_data.time_steps
        }
        
        # Convert to JSON and hash
        trajectory_json = json.dumps(trajectory_repr, sort_keys=True, separators=(',', ':'))
        trajectory_hash = hashlib.md5(trajectory_json.encode()).hexdigest()[:16]
        
        return trajectory_hash
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create hash of dictionary for cache key."""
        
        data_json = json.dumps(data, sort_keys=True, separators=(',', ':'), default=str)
        data_hash = hashlib.md5(data_json.encode()).hexdigest()[:8]
        
        return data_hash
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        
        # Try memory cache first (fastest)
        if self.memory_cache:
            result = await self.memory_cache.get(key)
            if result is not None:
                return result
        
        # Try Redis cache (if hybrid mode)
        if self.redis_cache and self.cache_type == "hybrid":
            result = await self.redis_cache.get(key)
            if result is not None:
                # Store in memory cache for faster future access
                if self.memory_cache:
                    await self.memory_cache.set(key, result, ttl=60)  # Short TTL for memory
                return result
        
        # Try Redis cache (if redis-only mode)
        elif self.redis_cache:
            return await self.redis_cache.get(key)
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        
        ttl = ttl or self.default_ttl
        
        # Set in memory cache
        if self.memory_cache:
            await self.memory_cache.set(key, value, ttl)
        
        # Set in Redis cache
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        
        results = []
        
        if self.memory_cache:
            result = await self.memory_cache.delete(key)
            results.append(result)
        
        if self.redis_cache:
            result = await self.redis_cache.delete(key)
            results.append(result)
        
        return any(results)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        
        if self.memory_cache:
            await self.memory_cache.clear()
        
        if self.redis_cache:
            await self.redis_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        stats = {
            "cache_type": self.cache_type,
            "default_ttl": self.default_ttl
        }
        
        if self.memory_cache:
            memory_stats = self.memory_cache.get_stats()
            stats["memory_cache"] = memory_stats
        
        if self.redis_cache:
            redis_stats = self.redis_cache.get_stats()
            stats["redis_cache"] = redis_stats
        
        # Combined statistics
        if self.memory_cache and self.redis_cache:
            total_hits = (stats["memory_cache"]["hit_count"] + 
                         stats["redis_cache"]["hit_count"])
            total_misses = (stats["memory_cache"]["miss_count"] + 
                           stats["redis_cache"]["miss_count"])
            total_requests = total_hits + total_misses
            
            stats["combined"] = {
                "total_requests": total_requests,
                "hit_count": total_hits,
                "miss_count": total_misses,
                "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0
            }
        
        # Calculate effective values
        if "memory_cache" in stats:
            effective_stats = stats["memory_cache"]
        elif "redis_cache" in stats:
            effective_stats = stats["redis_cache"]
        else:
            effective_stats = {"hit_rate": 0.0, "total_requests": 0}
        
        # Add top-level stats for API compatibility
        stats.update({
            "cache_size": effective_stats.get("cache_size", 0),
            "max_cache_size": effective_stats.get("max_cache_size", 0),
            "hit_count": effective_stats.get("hit_count", 0),
            "miss_count": effective_stats.get("miss_count", 0),
            "hit_rate": effective_stats.get("hit_rate", 0.0),
            "total_requests": effective_stats.get("total_requests", 0),
            "eviction_count": effective_stats.get("eviction_count", 0),
            "average_lookup_time_ms": 1.0  # Placeholder
        })
        
        return stats
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern."""
        
        # For memory cache, need to check all keys
        invalidated_count = 0
        
        if self.memory_cache:
            keys_to_delete = []
            for key in self.memory_cache._cache.keys():
                if self._matches_pattern(key, pattern):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                await self.memory_cache.delete(key)
                invalidated_count += 1
        
        if self.redis_cache:
            # Redis supports pattern matching natively
            try:
                if self.redis_cache.redis_client:
                    redis_pattern = self.redis_cache._make_key(pattern)
                    keys = await self.redis_cache.redis_client.keys(redis_pattern)
                    if keys:
                        await self.redis_cache.redis_client.delete(*keys)
                        invalidated_count += len(keys)
            except Exception as e:
                logger.error(f"Redis pattern invalidation failed: {e}")
        
        return invalidated_count
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard support)."""
        
        if "*" not in pattern:
            return key == pattern
        
        # Simple wildcard matching
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def warm_cache(self, predictions: List[Tuple[str, Any]]) -> int:
        """Warm cache with pre-computed predictions."""
        
        warmed_count = 0
        
        for cache_key, prediction in predictions:
            try:
                await self.set(cache_key, prediction)
                warmed_count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for key {cache_key}: {e}")
        
        logger.info(f"Warmed cache with {warmed_count} predictions")
        return warmed_count
    
    async def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health status."""
        
        health = {
            "status": "healthy",
            "cache_type": self.cache_type,
            "backends_available": []
        }
        
        if self.memory_cache:
            health["backends_available"].append("memory")
            memory_stats = self.memory_cache.get_stats()
            
            # Check memory cache health
            if memory_stats["cache_size"] >= memory_stats["max_cache_size"] * 0.9:
                health["memory_warning"] = "Memory cache nearly full"
        
        if self.redis_cache:
            try:
                if self.redis_cache.redis_client:
                    await self.redis_cache.redis_client.ping()
                    health["backends_available"].append("redis")
                else:
                    health["redis_warning"] = "Redis not connected"
            except Exception as e:
                health["redis_error"] = str(e)
                health["status"] = "degraded"
        
        if not health["backends_available"]:
            health["status"] = "unhealthy"
            health["error"] = "No cache backends available"
        
        return health