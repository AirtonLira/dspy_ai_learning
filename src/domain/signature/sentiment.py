import dspy

class SentimentSignature(dspy.Signature):
    """
    Classifica o sentimento de um texto.
    """
    text: str = dspy.InputField(desc="Texto a ser analisado")
    sentiment: str = dspy.OutputField(desc="positivo ou negativo")