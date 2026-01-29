import dspy
import os

class LLMConfig:
    _instance = None

    @classmethod
    def get_instance(cls, model="llama3", base_url="http://localhost:11434"):
        """Retorna a conexão com o LLM em memória (Singleton)."""
        if cls._instance is None:
            
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            llm_local_mode = os.getenv("DSPY_AI_LOCAL_MODE", "false").lower()
            if llm_local_mode == "true":
                llm = dspy.LM(
                    model="ollama/glm4:9b-chat-q3_K_M",
                    chat=True,
                    max_tokens=256,
                    local_mode=True
                )
                print("Usando modelo local (Ollama GLM4).")
            elif gemini_api_key:
                llm = dspy.LM(
                    model="gemini/gemini-1.5-flash",
                    api_key=gemini_api_key,
                    chat=True,
                    max_tokens=2048,
                    api_version="v1"
                )
                print("Usando modelo remoto (Google Gemini 1.5 Flash - v1).")
            else:
                llm = dspy.LM(
                    model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    chat=True,
                    max_tokens=256
                )
                print("Usando modelo remoto (Liquid LFM 2.5).")
            
            print(f"--- Inicializando conexao com Ollama ({model}) ---")
            cls._instance = llm
            dspy.settings.configure(lm=cls._instance)
        return cls._instance

def setup_llm():
    return LLMConfig.get_instance()

def get_data_path():
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(src_dir, 'domain', 'dataset', 'data', 'b2w_reviews.csv')