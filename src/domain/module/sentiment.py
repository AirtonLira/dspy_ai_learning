import dspy
from domain.signature.sentiment import SentimentSignature

class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SentimentSignature)

    def forward(self, text: str = None, **kwargs):
        if text is None and 'text' in kwargs:
            text = kwargs['text']
        return self.predict(text=text)