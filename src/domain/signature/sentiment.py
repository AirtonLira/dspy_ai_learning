import dspy


class SentimentSignature(dspy.Signature):
    """Classifica o sentimento de um texto de review em português"""
    
    text: str = dspy.InputField(desc="Texto do review do cliente")
    sentiment: str = dspy.OutputField(
        desc="Classifique como: positivo, negativo ou neutro"
    )


class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SentimentSignature)
    
    def forward(self, text: str = None, **kwargs):
        if text is None and 'text' in kwargs:
            text = kwargs['text']
        
        # Garantir que sempre retorna um Prediction válido
        try:
            result = self.predict(text=text)
            # Validar se o resultado tem o atributo sentiment
            if not hasattr(result, 'sentiment'):
                # Fallback: criar um Prediction manualmente
                return dspy.Prediction(sentiment="neutro")
            return result
        except Exception as e:
            print(f" Erro na predição: {e}")
            # Retornar prediction padrão em caso de erro
            return dspy.Prediction(sentiment="neutro")