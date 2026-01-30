import os
import dspy
from dspy.teleprompt import MIPROv2
from domain.module.sentiment import SentimentClassifier
from domain.evaluation.sentiment_eval import sentiment_dataset_train, sentiment_accuracy


from pathlib import Path
from domain.evaluation.logger import log_result
from dspy.evaluate import Evaluate

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "dspy_results.txt"


class SentimentMiproManager:
    def __init__(self):
        
        dataset = sentiment_dataset_train()
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


    def run_mipro_optimization(self, num_candidates=3):
        # Carregar configurações de throttling/execução via env vars (podem ser ajustadas)
        dry_run = os.getenv("DSPY_MIPRO_DRY_RUN", "false").lower() == "true"

        print(f"MIPROv2 config -> dry_run={dry_run}, rate_limit: 1 req/60s")

        if dry_run:
            print("Dry-run ativado: pulando otimização pesada do MIPROv2.")
            return None

        # MIPROv2 com configuração minimalista para reduzir chamadas à API
        teleprompter = MIPROv2(
                prompt_model=dspy.settings.lm,
                task_model=dspy.settings.lm,
                metric=self._metric,
                auto="light"
            )

        compiled_program = teleprompter.compile(
                self.base_program,
                trainset=self.trainset,
                max_bootstrapped_demos=0,
                max_labeled_demos=0
        )
        
        
        self.compiled_program = compiled_program

        print("\n Avaliando programa otimizado...")
        
        # Criar testset (últimos 10% do dataset)
        split_point = int(len(self.trainset) * 0.9)
        testset = self.trainset[split_point:]
        
        # Avaliar
        evaluator = Evaluate(
            devset=testset,
            metric=self._metric,
            display_progress=True,
            display_table=False
        )


        
        result_score = evaluator(self.compiled_program)
        
        #Acessar o score dentro do resultado
        if hasattr(result_score, 'score'):
            score = result_score.score
        elif isinstance(result_score, (int, float)):
            score = result_score
        else:
            # Fallback: converter para float
            score = float(result_score)
        
        
        print(f"\n Acurácia no testset: {score:.2%}")
        
        # Mostrar exemplo prático
        self._show_example_prediction()
        
    def save_checkpoint(self, filename="sentiment_mipro_final.json"):
        if self.compiled_program:
            self.compiled_program.save(RESULTS_DIR / filename)
                
    def _show_example_prediction(self):
        """Mostra um exemplo de predição"""
        if not self.compiled_program or not self.trainset:
            return
        
        example = self.trainset[0]
        
        print("\n EXEMPLO DE PREDIÇÃO:")
        print("="*60)
        print(f"Texto: {example.text}")
        
        # 
        prediction = self.compiled_program(text=example.text)
        
        print(f"Sentimento real: {example.sentiment}")
        print(f"Sentimento previsto: {prediction.sentiment}")
        
        match = "CORRETO" if example.sentiment.lower() == prediction.sentiment.lower() else "❌ ERRADO"
        print(f"Resultado: {match}")
        print("="*60)
        
        scores = []
        for example in self.trainset:
            prediction = self.compiled_program(text=example.text)
            score = sentiment_accuracy(example, prediction)
            scores.append(score)

        print("\n Avaliando acurácia no conjunto de treino completo...")  
        print("="*60)
        accuracy = sum(scores) / len(scores)
        log_result(
            phase="MIPROv2_evaluation_optimized",
            metric_name="accuracy",
            metric_value=accuracy,
            num_examples=len(scores),
            model_name="gemini/gemini-3-flash-preview",
            notes="baseline"
        )
        print(f"Acurácia final (otimizada): {accuracy:.2f}")