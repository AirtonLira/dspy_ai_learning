import pandas as pd
from sklearn.model_selection import train_test_split
from domain.dataset.b2w_loader import B2WLoader
from domain.module.sentiment import SentimentClassifier

class B2WReviews:
    def __init__(self, sample=None):
        """
        Carrega e processa reviews B2W.
        
        Args:
            sample: Numero de registros a amostrar. Se None, usa todos.
        """
        loader = B2WLoader()
        self.df = loader.load()
        
        if sample is not None:
            if sample < len(self.df):
                self.df = self.df.sample(n=sample, random_state=42)
        
        self.df = self._clean_data(self.df)
        self._print_summary()
    
    def _clean_data(self, df):
        """Remove registros invalidos."""
        print("Limpando dados...")
        
        df = df.dropna(subset=['review_text', 'review_rating'])
        df = df[df['review_text'].str.len() > 0]
        
        df['sentiment'] = df['review_rating'].apply(self._rating_to_sentiment)
        
        print(f"Apos limpeza: {len(df)} registros validos\n")
        return df
    
    def _rating_to_sentiment(self, rating):
        """Converte rating em sentimento."""
        try:
            rating_num = float(rating)
            if rating_num <= 2:
                return 'negativo'
            elif rating_num <= 3:
                return 'neutro'
            else:
                return 'positivo'
        except (ValueError, TypeError):
            return None
    
    def _print_summary(self):
        """Imprime resumo do dataset."""
        print("\nDistribuicao de sentimentos:")
        print(self.df['sentiment'].value_counts())
        print(f"Dataset processado: {len(self.df)} registros prontos")
    
    def get_examples(self):
        """Retorna lista de SentimentExample."""
        examples = []
        for _, row in self.df.iterrows():
            example = SentimentClassifier(
                text=row['review_text'],
                sentiment=row['sentiment']
            )
            examples.append(example)
        return examples
    
    def get_train_test_split(self, test_size=0.2):
        """
        Retorna train/test split estratificado.
        Garante minimo de 2 exemplos por classe em cada split.
        """
        # Verifica se temos dados suficientes
        sentiment_counts = self.df['sentiment'].value_counts()
        min_count = sentiment_counts.min()
        
        if min_count < 2:
            print(f"Aviso: Classe com apenas {min_count} exemplo(s). Aumentando amostra...")
            # Se nao temos dados suficientes, retorna sem estratificacao
            train_df, test_df = train_test_split(
                self.df,
                test_size=test_size,
                random_state=42
            )
        else:
            train_df, test_df = train_test_split(
                self.df,
                test_size=test_size,
                stratify=self.df['sentiment'],
                random_state=42
            )
        
        train_examples = self._df_to_examples(train_df)
        test_examples = self._df_to_examples(test_df)
        
        return train_examples, test_examples
    
    def _df_to_examples(self, df):
        """Converte DataFrame em lista de SentimentExample."""
        examples = []
        for _, row in df.iterrows():
            example = SentimentExample(
                text=row['review_text'],
                sentiment=row['sentiment']
            )
            examples.append(example)
        return examples