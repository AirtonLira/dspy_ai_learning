import os
import pandas as pd
from dspy import Example
from tqdm import tqdm

from utils.config import get_data_path

class B2WReviews:
    def __init__(self, path: str = None, sample: int = None, train_size: float = 0.8):
        self.path = path if path else get_data_path()
        self.sample = sample
        self.train_size = train_size
        self._load_data()

    def _load_data(self) -> None:
        """Loads the B2W reviews dataset from a CSV file."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The file {self.path} was not found.")

        self.df = pd.read_csv(self.path)

        if self.sample:
            self.df = self.df.sample(n=self.sample, random_state=42)

        self.df = self.df.dropna(subset=['review_text', 'overall_rating'])
        self.df = self.df[['review_text', 'overall_rating']]
        self.df['sentiment'] = self.df['overall_rating'].apply(self._classify_sentiment)
        self.df = self.df.rename(columns={'review_text': 'text'})

    @staticmethod
    def _classify_sentiment(rating: float) -> str:
        """Classifies sentiment based on the overall rating."""
        if rating > 3:
            return 'positivo'
        elif rating < 3:
            return 'negativo'
        return 'neutro'

    def _format_for_dspy(self, df: pd.DataFrame) -> list[Example]:
        """Formats a DataFrame into a list of dspy.Example objects."""
        formatted_examples = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Formatting examples"):
            example = Example(
                text=row['text'],
                sentiment=row['sentiment']
            ).with_inputs("text")
            formatted_examples.append(example)
        return formatted_examples

    def get_train_test_split(self) -> tuple[list[Example], list[Example]]:
        """Splits the data into training and testing sets and formats them for dspy."""
        train_df = self.df.sample(frac=self.train_size, random_state=42)
        test_df = self.df.drop(train_df.index)

        train_set = self._format_for_dspy(train_df)
        test_set = self._format_for_dspy(test_df)

        return train_set, test_set
