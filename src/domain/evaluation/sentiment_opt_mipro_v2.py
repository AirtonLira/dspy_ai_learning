import dspy
from dspy.teleprompt import MIPROv2
from domain.module.sentiment import SentimentClassifier
from utils.rate_limiter import rate_limiter
from domain.evaluation.sentiment_eval import (
    sentiment_dataset,
)
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR 


class SentimentMiproManager:
    def __init__(self):
        
        dataset = sentiment_dataset()
        self.base_program = SentimentClassifier()

        if not dataset:
            print("Erro: Dataset vazio!")
            return None
        
        self.trainset = dataset
        # Instanciando sua classe real do repositório
        self.compiled_program = None

    def _metric(self, example, pred, trace=None):
        # Baseado nos campos sentiment e answer definidos nas assinaturas
        return example.sentiment.lower() == pred.sentiment.lower()


    @rate_limiter
    def run_mipro_optimization(self,num_candidates=3):
        
        teleprompter = MIPROv2(
                prompt_model=dspy.settings.lm, 
                task_model=dspy.settings.lm, 
                metric=self._metric, 
                auto="medium"
            )
        
        compiled_program = teleprompter.compile(
                self.base_program,
                trainset=self.trainset,
                max_bootstrapped_demos=2,
                max_labeled_demos=2
        )
        self.compiled_program = compiled_program

    def save_checkpoint(self, filename="sentiment_mipro_final.json"):
            if self.compiled_program:
                self.compiled_program.save(RESULTS_FILE.parent / filename)