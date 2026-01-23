import os
from time import sleep
from domain.module.sentiment import SentimentClassifier
from domain.dataset.b2w_review import load_b2w_reviews
from domain.evaluation.logger import log_result

# Dataset de avaliação
def sentiment_dataset():
    return load_b2w_reviews(limit=int(os.getenv("LIMIT_DATASET_EVAL")))


# Métrica de avaliação
def sentiment_accuracy(example, prediction, trace=None):
    """
    Retorna 1 se acertou, 0 se errou.
    """
    return int(
        example.sentiment.strip().lower()
        == prediction.sentiment.strip().lower()
    )


def run_evaluation():
    classifier = SentimentClassifier()
    dataset = sentiment_dataset()

    scores = []
    print("Iniciando avaliação de sentimento...")
    for example in dataset:
        prediction = classifier(text=example.text)
        score = sentiment_accuracy(example, prediction)
        print(f"Texto: {example.text}, Esperado: {example.sentiment}, Predito: {prediction.sentiment}, Score: {score}")
        scores.append(score)
        sleep(8)  # Pequena pausa para evitar sobrecarga

        # print("Texto:", example.text)
        # print("Esperado:", example.sentiment)
        # print("Predito :", prediction.sentiment)
        # print("Score   :", score)
        # print("-" * 50)

    accuracy = sum(scores) / len(scores)
    
    log_result(
        phase="evaluation",
        metric_name="accuracy",
        metric_value=accuracy,
        num_examples=len(scores),
        model_name="ollama/llama3.1",
        notes="baseline"
    )
    print(f"Acurácia final: {accuracy:.2f}")
