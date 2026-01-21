
import dspy
from domain.module.sentiment import SentimentClassifier



llm = dspy.LM(
    model="ollama/smollm2:1.7b",
    max_tokens=256,
    chat=True  
)


dspy.settings.configure(lm=llm)


if __name__ == "__main__":
    classifier = SentimentClassifier()

    examples = [
        "Eu amo este produto, é incrível!",
        "Esta é a pior experiência que já tive.",
        "O serviço foi ok, nada de especial.",
        "Equipe de suporte absolutamente fantástica!",
        "Eu odeio isso, total desperdício de dinheiro."
    ]

    for text in examples:
        result = classifier(text=text)
        print(f"Texto: {text}")
        print(f"Sentimento: {result.sentiment}")
        print("-" * 50)
