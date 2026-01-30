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


    def run_mipro_optimization(self, num_candidates: int = 3):
        """Executa otimização MIPRO"""
        
        # Verificar se o LM foi configurado
        if not dspy.settings.lm:
            raise ValueError("LM não configurado! Chame setup_llm() primeiro.")
        
        print(f"\n{'='*60}")
        
        # Configurar MIPRO com os modelos explícitos
        teleprompter = MIPROv2(
            metric=self._metric,
            prompt_model=dspy.settings.lm,  # Modelo para gerar prompts
            task_model=dspy.settings.lm,    # Modelo para executar tarefas
            auto="medium"
        )
        
        # Compilar programa
        compiled_program = teleprompter.compile(
            self.base_program,
            trainset=self.trainset,
            max_bootstrapped_demos=0,
            max_labeled_demos=0
        )
        
        self.compiled_program = compiled_program
        
        # Avaliar
        print("\n Avaliando programa otimizado...")
        evaluator = Evaluate(
            devset=self.testset,
            metric=self._metric,
            num_threads=4,
            display_progress=True,
            display_table=False,
        )
        
        accuracy = evaluator(self.compiled_program)
        print(f"\n Acurácia no testset: {accuracy:.2%}\n")
        
        # Mostrar exemplo
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