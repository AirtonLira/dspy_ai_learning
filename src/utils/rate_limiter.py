import time
import functools
from collections import deque
from datetime import datetime, timedelta
import threading


class RateLimiter:
    """
    Rate limiter para Gemini API (free tier: 5 req/min).
    
    Implementa:
    - Janela deslizante (sliding window)
    - Retry automático com exponential backoff
    - Thread-safe
    
    Uso:
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        @limiter
        def api_call():
            return gemini.generate(...)
    """
    
    def __init__(self, max_requests=5, window_seconds=60):
        """
        Args:
            max_requests: Máximo de requisições na janela (padrão: 5)
            window_seconds: Tamanho da janela em segundos (padrão: 60s = 1 min)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()  # Fila de timestamps
        self.lock = threading.Lock()
        print(f"\n RateLimiter inicializado: {max_requests} req/{window_seconds}s")
    
    def wait_if_needed(self) -> float:
        """
        Aguarda se necessário para não exceder rate limit.
        
        Returns:
            float: Segundos aguardados (0 se não teve espera)
        """
        with self.lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            # Remover requisições fora da janela de tempo
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            
            # Se atingimos o limite máximo, aguardar até poder fazer nova requisição
            if len(self.requests) >= self.max_requests:
                wait_time = (self.requests[0] + timedelta(seconds=self.window_seconds) - now).total_seconds()
                
                if wait_time > 0:
                    print(f"Rate limit. Aguardando {wait_time:.1f}s...")
                    time.sleep(wait_time + 0.1)  # +0.1s de margem
                    return wait_time
            
            # Registrar esta requisição
            self.requests.append(now)
            return 0.0
    
    def __call__(self, func):
        """
        Torna a classe usável como decorador.
        Implementa retry automático com exponential backoff.
        
        Args:
            func: Função a decorar
            
        Returns:
            Função decorada com rate limit e retry
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = 5
            retry_delay = 4  #
            
            for attempt in range(1, max_retries + 1):
                try:
                    # Aguardar se necessário
                    self.wait_if_needed()
                    
                    # Executar função
                    return func(*args, **kwargs)
                
                except Exception as e:
                    error_msg = str(e)
                    
                    # Detectar se é erro de rate limit
                    is_rate_limit = (
                        "429" in error_msg or 
                        "quota" in error_msg.lower() or
                        "RESOURCE_EXHAUSTED" in error_msg or
                        "RateLimitError" in str(type(e).__name__)
                    )
                    
                    # Retry se for rate limit e ainda temos tentativas
                    if is_rate_limit and attempt < max_retries:
                        print(f"\n Tentativa {attempt}/{max_retries}: Rate limit detectado")
                        print(f"   Aguardando {retry_delay}s antes do retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff: 2, 4, 8, 16...
                        continue
                    
                    # Se chegou na última tentativa, relançar o erro
                    if attempt == max_retries:
                        print(f"\n❌ Falha após {max_retries} tentativas")
                    
                    raise
        
        return wrapper


# Singleton global para usar em qualquer lugar
gemini_rate_limiter = RateLimiter(max_requests=3, window_seconds=60)