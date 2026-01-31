import os
import random # Importado para embaralhar os dados
import dspy
from dspy.teleprompt import MIPROv2
from domain.module.sentiment import SentimentClassifier
from domain.evaluation.sentiment_eval import sentiment_dataset_train, sentiment_accuracy

from pathlib import Path
from domain.evaluation.logger import log_result
from dspy.evaluate import Evaluate

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

class SentimentMiproManager:
    def __init__(self, train_size=0.8): # Adicionado parâmetro de proporção
        
        full_dataset = sentiment_dataset_train()
        self.base_program = SentimentClassifier()

        if not full_dataset:
            print("Erro: Dataset vazio!")
            return
        
        # --- SEÇÃO DE SEPARAÇÃO (SPLIT) ---
        # Embaralhamos para garantir que a distribuição de classes seja aleatória
        random.seed(42) 
        random.shuffle(full_dataset)
        
        split_idx = int(len(full_dataset) * train_size)
        self.trainset = full_dataset[:split_idx]  # Usado para compilar/otimizar
        self.testset = full_dataset[split_idx:]   # Usado para avaliação final
        # ----------------------------------

        print(f"Dataset carregado: {len(self.trainset)} treino / {len(self.testset)} teste")
        self.compiled_program = None

    def _metric(self, example, pred, trace=None):
        return example.sentiment.lower() == pred.sentiment.lower()

    def run_mipro_optimization(self, num_candidates: int = 2):
        if not dspy.settings.lm:
            raise ValueError("LM não configurado! Chame setup_llm() primeiro.")
        
        print(f"\n{'='*60}")
        print(f"Iniciando Otimização MIPROv2 com {len(self.trainset)} exemplos...")
        
        teleprompter = MIPROv2(
            metric=self._metric,
            prompt_model=dspy.settings.lm,
            task_model=dspy.settings.lm,
            auto="light",
            num_threads=1,
            verbose=False
        )
        
        # O MIPRO usa o trainset para criar os prompts e demonstrações
        self.compiled_program = teleprompter.compile(
            self.base_program,
            trainset=self.trainset,
            max_bootstrapped_demos=2,
            max_labeled_demos=2
        )
        
        # Avaliar no TESTSET (dados que o otimizador nunca viu)
        print("\n--- Avaliando no CONJUNTO DE TESTE (Inédito) ---")
        evaluator = Evaluate(
            devset=self.testset, # Aqui usamos o conjunto de teste separado no __init__
            metric=self._metric,
            num_threads=1,
            display_progress=True,
            display_table=False,
        )
        
        test_accuracy = float(evaluator(self.compiled_program))
        print(f"\n Acurácia Final no Testset: {test_accuracy:.2%}\n")
        
        self._log_final_results(test_accuracy)

    def _log_final_results(self, accuracy): 
        log_result(
            phase="MIPROv2_evaluation",
            metric_name="accuracy",
            metric_value=accuracy,
            num_examples=len(self.testset),
            model_name="gemini/gemini-1.5-flash", # Ajuste conforme seu setup
            notes="Avaliação em testset separado"
        )

    def save_checkpoint(self, filename="sentiment_mipro_final.json"):
        if self.compiled_program:
            self.compiled_program.save(RESULTS_DIR / filename)