import time
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class RateLimitCallback:
    """Callback para rate limiting sem wrapper no LM"""
    
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    def before_call(self, **kwargs: Any) -> None:
        """Executado antes de cada chamada ao LM"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            logger.debug(f"Rate limit: aguardando {wait_time:.2f}s")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
