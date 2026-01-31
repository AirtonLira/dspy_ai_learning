import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.evaluation.sentiment_eval import run_evaluation
from dotenv import load_dotenv
from domain.evaluation.sentiment_opt_fewshot import run_optimization
from domain.evaluation.sentiment_opt_mipro_v2 import SentimentMiproManager
from utils.config import setup_llm

# Suprimir TODOS os warnings do console
warnings.filterwarnings('ignore')

# Funções de fluxo do miprov2
def run_mipro_flow():
    """Encapsula a sequência de comandos do MIPROv2"""
    # O MIPRO é mais pesado pois propõe candidatos e analisa o dataset
    manager = SentimentMiproManager()
    manager.run_mipro_optimization(num_candidates=3)
    manager.save_checkpoint("sentiment_mipro_final.json")


def main():
    load_dotenv()
    setup_llm()

    # 1. Mapeamento de Opções
    strategies = {
        "EVALUATION": run_evaluation,
        "BOOTSTRAP": run_optimization,
        "MIPRO": run_mipro_flow
        }
    
    # 3. Execução
    choice = os.getenv("OPTIMIZER_TYPE", "MIPRO").upper()
    
    # Busca a função no dicionário. Se não achar, usa run_evaluation como padrão.
    action = strategies.get(choice, run_evaluation)
    
    print(f"--- Iniciando: {choice} ---")
    
    # AGORA sim nós executamos a função escolhida
    action()
    
if __name__ == "__main__":
    main()