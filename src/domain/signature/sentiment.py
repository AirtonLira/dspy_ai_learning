import dspy

class SentimentSignature(dspy.Signature):
    """Classifique o sentimento do review em: positivo, negativo ou neutro."""
    text: str = dspy.InputField(desc="Review de produto em português")
    sentiment: Literal["positivo", "neutro", "negativo"] = dspy.OutputField(
        desc="Sentimento do review: 'positivo', 'neutro' ou 'negativo'"
    )


class SentimentClassifier(dspy.Module):
    """Módulo de classificação de sentimento usando DSPy."""

    def __init__(self):
        super().__init__()
        # Você pode trocar dspy.Predict por um módulo mais sofisticado (ChainOfThought, etc.)
        self.predict = dspy.Predict(SentimentSignature)

    def forward(self, text: str) -> dspy.Prediction:
        """Faz a predição de sentimento, com fallback em caso de AdapterParseError."""
        try:
            prediction = self.predict(text=text)
            # Garantir que o campo esteja presente
            if not getattr(prediction, "sentiment", None):
                # fallback simples; você pode logar ou re-tentar aqui
                return dspy.Prediction(sentiment="neutro")
            return prediction
        except AdapterParseError as e:
            # Evita crash quando o JSONAdapter não consegue parsear o LM
            print(f"[WARN] Falha ao parsear resposta do LM: {e}")
            # fallback conservador
            return dspy.Prediction(sentiment="neutro")