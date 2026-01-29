import dspy

class SentimentSignature(dspy.Signature):
    """
    Classifica o sentimento de um texto. Responda APENAS com a palavra 'positivo' ou 'negativo'. Não inclua explicações ou texto adicional.
    """
    text: str = dspy.InputField(desc="Texto a ser analisado")
    sentiment: str = dspy.OutputField(desc="Categoria de sentimento: 'positivo' ou 'negativo'")