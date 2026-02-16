# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Model metadata cache for Kiro Gateway.

Thread-safe storage for available model information
with TTL, lazy loading support, and persistent JSON storage.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.config import MODEL_CACHE_TTL, DEFAULT_MAX_INPUT_TOKENS


class ModelInfoCache:
    """
    Thread-safe cache for storing model metadata.
    
    Uses Lazy Loading for population - data is loaded
    only on first access or when cache is stale.
    
    Persists to cache.json to avoid fetching from API on every startup.
    
    Attributes:
        cache_ttl: Cache time-to-live in seconds
        cache_file: Path to cache.json file
    
    Example:
        >>> cache = ModelInfoCache()
        >>> await cache.update([{"modelId": "claude-sonnet-4", "tokenLimits": {...}}])
        >>> info = cache.get("claude-sonnet-4")
        >>> max_tokens = cache.get_max_input_tokens("claude-sonnet-4")
    """
    
    def __init__(self, cache_ttl: int = MODEL_CACHE_TTL, cache_file: str = "cache.json"):
        """
        Initializes the model cache.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default from config)
            cache_file: Path to cache file (default: cache.json)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_update: Optional[float] = None
        self._cache_ttl = cache_ttl
        self._cache_file = Path(cache_file)
        
        # Try to load from cache file on initialization
        self._load_from_file()
    
    def _load_from_file(self) -> None:
        """
        Load cache from cache.json file (synchronous, called during __init__).
        
        If file doesn't exist or is invalid, silently continues with empty cache.
        Checks if cache is stale and logs appropriate message.
        """
        if not self._cache_file.exists():
            logger.debug(f"Cache file not found: {self._cache_file}")
            return
        
        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._cache = data.get("models", {})
            self._last_update = data.get("last_update")
            
            if self._last_update:
                age_seconds = time.time() - self._last_update
                age_hours = age_seconds / 3600
                
                if self.is_stale():
                    logger.info(f"Loaded stale cache from {self._cache_file} ({len(self._cache)} models, {age_hours:.1f}h old). Will refresh from API.")
                else:
                    logger.info(f"Loaded fresh cache from {self._cache_file} ({len(self._cache)} models, {age_hours:.1f}h old)")
            else:
                logger.info(f"Loaded cache from {self._cache_file} ({len(self._cache)} models, no timestamp)")
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse cache file {self._cache_file}: {e}. Starting with empty cache.")
        except Exception as e:
            logger.warning(f"Failed to load cache from {self._cache_file}: {e}. Starting with empty cache.")
    
    def _save_to_file(self) -> None:
        """
        Save cache to cache.json file (synchronous).
        
        Creates the file if it doesn't exist.
        Logs errors but doesn't raise exceptions (cache persistence is optional).
        """
        try:
            data = {
                "models": self._cache,
                "last_update": self._last_update,
                "cache_ttl": self._cache_ttl,
            }
            
            # Write atomically: write to temp file, then rename
            temp_file = self._cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename (overwrites existing file)
            temp_file.replace(self._cache_file)
            logger.debug(f"Saved cache to {self._cache_file} ({len(self._cache)} models)")
        
        except Exception as e:
            logger.warning(f"Failed to save cache to {self._cache_file}: {e}")
    
    async def update(self, models_data: List[Dict[str, Any]]) -> None:
        """
        Updates the model cache and persists to cache.json.
        
        Thread-safely replaces cache contents with new data.
        
        Args:
            models_data: List of dictionaries with model information.
                        Each dictionary must contain the "modelId" key.
        """
        async with self._lock:
            logger.info(f"Updating model cache. Found {len(models_data)} models.")
            self._cache = {model["modelId"]: model for model in models_data}
            self._last_update = time.time()
            
            # Save to file
            self._save_to_file()
    
    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns model information.
        
        Args:
            model_id: Model ID
        
        Returns:
            Dictionary with model information or None if model not found
        """
        return self._cache.get(model_id)
    
    def is_valid_model(self, model_id: str) -> bool:
        """
        Check if model exists in dynamic cache.
        
        Used by ModelResolver to verify if a model is available.
        
        Args:
            model_id: Model ID to check
        
        Returns:
            True if model exists in cache, False otherwise
        """
        return model_id in self._cache
    
    def add_hidden_model(self, display_name: str, internal_id: str) -> None:
        """
        Add a hidden model to the cache.
        
        Hidden models are not returned by Kiro /ListAvailableModels API
        but are still functional. They are added to the cache so they
        appear in our /v1/models endpoint.
        
        Args:
            display_name: Model name to display (e.g., "claude-3.7-sonnet")
            internal_id: Internal Kiro ID (e.g., "CLAUDE_3_7_SONNET_20250219_V1_0")
        """
        if display_name not in self._cache:
            self._cache[display_name] = {
                "modelId": display_name,
                "modelName": display_name,
                "description": f"Hidden model (internal: {internal_id})",
                "tokenLimits": {"maxInputTokens": DEFAULT_MAX_INPUT_TOKENS},
                "_internal_id": internal_id,  # Store internal ID for reference
                "_is_hidden": True,  # Mark as hidden model
            }
            logger.debug(f"Added hidden model: {display_name} → {internal_id}")
    
    def get_max_input_tokens(self, model_id: str) -> int:
        """
        Returns maxInputTokens for the model.
        
        Args:
            model_id: Model ID
        
        Returns:
            Maximum number of input tokens or DEFAULT_MAX_INPUT_TOKENS
        """
        model = self._cache.get(model_id)
        if model and model.get("tokenLimits"):
            return model["tokenLimits"].get("maxInputTokens") or DEFAULT_MAX_INPUT_TOKENS
        return DEFAULT_MAX_INPUT_TOKENS
    
    def is_empty(self) -> bool:
        """
        Checks if the cache is empty.
        
        Returns:
            True if cache is empty
        """
        return not self._cache
    
    def is_stale(self) -> bool:
        """
        Checks if the cache is stale.
        
        Returns:
            True if cache is stale (more than cache_ttl seconds have passed)
            or if cache was never updated
        """
        if not self._last_update:
            return True
        return time.time() - self._last_update > self._cache_ttl
    
    def get_all_model_ids(self) -> List[str]:
        """
        Returns a list of all model IDs in the cache.
        
        Returns:
            List of model IDs
        """
        return list(self._cache.keys())
    
    @property
    def size(self) -> int:
        """Number of models in the cache."""
        return len(self._cache)
    
    @property
    def last_update_time(self) -> Optional[float]:
        """Last update time (timestamp) or None."""
        return self._last_update