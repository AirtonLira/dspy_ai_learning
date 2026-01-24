import time
import functools
from collections import deque
from datetime import datetime, timedelta
import threading


class RateLimiter:
    """Rate limiter para Gemini API (max_requests req/window_seconds segundos)."""
    
    def __init__(self, max_requests=5, window_seconds=30):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()
        
    def wait_if_needed(self):
        """Aguarda se necessário para não exceder rate limit."""
        with self.lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            # Remover requisições fora da janela
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            
            # Se atingimos limite, aguardar
            if len(self.requests) >= self.max_requests:
                wait_time = (self.requests[0] + timedelta(seconds=self.window_seconds) - now).total_seconds()
                if wait_time > 0:
                    print(f"Rate limit. Aguardando {wait_time:.1f}s...")
                    time.sleep(wait_time + 0.1)  # +0.1s de margem
                    return wait_time
            
            self.requests.append(now)
            return 0.0
        
        def __call__(self, func):
            """Decorador com retry automático."""
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                max_retries = 5
                retry_delay = 2
                
                for attempt in range(1, max_retries + 1):
                    try:
                        self.wait_if_needed()
                        return func(*args, **kwargs)
                    
                    except Exception as e:
                        error_msg = str(e)
                        is_rate_limit = "429" in error_msg or "quota" in error_msg.lower()
                        
                        if is_rate_limit and attempt < max_retries:
                            print(f"Tentativa {attempt}: Rate limit. Retry em {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        
                        raise
            
            return wrapper
    
# Singleton global
rate_limiter = RateLimiter(max_requests=5, window_seconds=30)