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
   
       print("Texto:", example.text)
       print("Esperado:", example.sentiment)
       print("Predito :", prediction.sentiment)
       print("Score   :", score)
       print("-" * 50)
       
   accuracy = sum(scores) / len(scores)
   log_result(
        phase="evaluation_optimized",
        metric_name="accuracy_fewshot",
        metric_value=accuracy,
        num_examples=len(scores),
        model_name="ollama/llama3.1",
        notes="baseline"
    )   
   print(f"Acurácia final (otimizada): {accuracy:.2f}")
   
   print("\n=== Exemplos Selecionados pelo Otimizador ===\n")

   # O SentimentClassifier geralmente tem um predictor interno (ou ChainOfThought)
   # Acessamos o predictor para ver os exemplos que ele 'aprendeu'
   for i, demo in enumerate(optimized_program.predict.demos):
       print(f"Exemplo {i+1}:")
       print(f"Texto: {demo.text}")
       print(f"Sentimento: {demo.sentiment}")
       print("-" * 20)
       
   print("\n=== Exemplo de Prompt enviado ao LLM ===\n")
   
   
   # Acessa todos os prompts feitos desde que o script começou
   history = dspy.settings.lm.history  
   
   print(f"O otimizador realizou {len(history)} chamadas ao modelo.")  
   # Ver o primeiro prompt que ele tentou no treino
   if history:
     # Acessa a primeira escolha, a mensagem e o conteúdo textual
     resposta_bruta = history[0]['response']
        
     # Verifica se é o objeto ModelResponse e extrai o conteúdo
     if hasattr(resposta_bruta, 'choices'):
         conteúdo = resposta_bruta.choices[0].message.content
         print("Resposta formatada:\n", conteúdo)
     else:
         print("Resposta:", resposta_bruta)
    
   return optimized_program