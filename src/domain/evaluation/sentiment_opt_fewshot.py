import dspy
from domain.module.sentiment import SentimentClassifier
from domain.evaluation.sentiment_eval import (
    sentiment_dataset_train,
    sentiment_accuracy,
)
from domain.evaluation.logger import log_result

def run_optimization():
    
   # Pipeline original
   base_program = SentimentClassifier()
   dataset = sentiment_dataset_train()
   
   if not dataset:
       print("Erro: Dataset vazio!")
       return base_program
   
   # Otimizador
   optimzer = dspy.BootstrapFewShot(
       metric=sentiment_accuracy,
       max_bootstrapped_demos=4
   )
   
   # Treinamento com otimização
   optimized_program = optimzer.compile(
       base_program,
       trainset=dataset
   )
   
   print("\n=== Resultados da Otimização ===\n")
   
   scores = []
   
   for example in dataset:
       prediction = optimized_program(text=example.text)
       score = sentiment_accuracy(example, prediction)
       scores.append(score)
   
    #    print("Texto:", example.text)
    #    print("Esperado:", example.sentiment)
    #    print("Predito :", prediction.sentiment)
    #    print("Score   :", score)
    #    print("-" * 50)
       
   accuracy = sum(scores) / len(scores)
   log_result(
        phase="evaluation_optimized",
        metric_name="accuracy",
        metric_value=accuracy,
        num_examples=len(scores),
        model_name="ollama/llama3.1",
        notes="baseline"
    )
   print(f"Acurácia final (otimizada): {accuracy:.2f}")
    
   return optimized_program