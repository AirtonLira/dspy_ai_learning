import dspy
import os
from utils.rate_limiter import gemini_rate_limiter

class LLMConfig:
    _instance = None

    @classmethod
    def get_instance(cls, model="llama3", base_url="http://localhost:11434"):
        """Retorna a conexão com o LLM em memória (Singleton)."""
        if cls._instance is None:
            
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            llm_local_mode = os.getenv("DSPY_AI_LOCAL_MODE").lower()
            if llm_local_mode == "true":
                llm = dspy.LM(
                    model="ollama/llama3.2:1b",
                    chat=True,
                    max_tokens=20, # Reduzido para evitar conversas longas
                    temperature=0.0, # Mais determinístico
                    local_mode=True
                )
                print("Usando modelo local (Ollama Llama3.2).")
            elif gemini_api_key:
                llm = dspy.LM(
                    model="gemini/gemini-3-flash-preview",
                    api_key=gemini_api_key,
                    chat=True,
                    max_tokens=2048
                )
                print("Usando modelo remoto (Google Gemini).")
            elif llm_local_mode == "false":
                llm = dspy.LM(
                    model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    chat=True,
                    max_tokens=256
                )
                print("Usando modelo remoto (Liquid LFM 2.5).")

            
            # Aplicar Rate Limit diretamente no método __call__ do LM
            # Isso garante que todas as chamadas do DSPy passem pelo limitador
            original_call = llm.__call__
            
            @gemini_rate_limiter
            def rate_limited_call(*args, **kwargs):
                return original_call(*args, **kwargs)
            
            llm.__call__ = rate_limited_call
            
            cls._instance = llm
            dspy.settings.configure(lm=cls._instance)
        return cls._instance

# Atalho para facilitar o uso
def setup_llm():
    return LLMConfig.get_instance()

def get_data_path():
    # The path to the src directory
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(src_dir, 'domain', 'dataset', 'data', 'b2w_reviews.csv')