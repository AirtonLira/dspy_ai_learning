import dspy
from domain.module.sentiment import SentimentClassifier


# Dataset de avaliação
def sentiment_dataset():
    return [
        dspy.Example(
            text="Eu amo este produto, é incrível!",
            sentiment="positivo"
        ),
        dspy.Example(
            text="Esta é a pior experiência que já tive.",
            sentiment="negativo"
        ),
        dspy.Example(
            text="O serviço foi ok, nada de especial.",
            sentiment="negativo"
        ),
        dspy.Example(
            text="Equipe de suporte absolutamente fantástica!",
            sentiment="positivo"
        ),
        dspy.Example(
            text="Eu odeio isso, total desperdício de dinheiro.",
            sentiment="negativo"
        ),
    ]


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

    for example in dataset:
        prediction = classifier(text=example.text)
        score = sentiment_accuracy(example, prediction)
        scores.append(score)

        print("Texto:", example.text)
        print("Esperado:", example.sentiment)
        print("Predito :", prediction.sentiment)
        print("Score   :", score)
        print("-" * 50)

    accuracy = sum(scores) / len(scores)
    print(f"Acurácia final: {accuracy:.2f}")
