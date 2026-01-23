from dspy import Example
import csv
from pathlib import Path

import dspy
from domain.dataset.schema import B2WRawschema, DSPyReviewSchema
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "b2w_reviews.csv"



def map_rating_to_sentiment(rating: int) -> str | None:
    """
    Faz o mapeamento de rating numérico para sentimento textual.
    """
    if rating >= 4:
        return "Positivo"
    if rating <= 2:
        return "Negativo"
    return None

def b2w_review(row: dict) -> dspy.Example | None:
    """
    Converte uma linha do CSV em um dspy.Example normalizado.
    """
    try:
        text = row[B2WRawschema.REVIEW_TEXT].strip()
        rating = int(row[B2WRawschema.OVERALL_RATING])
    except (KeyError, ValueError, AttributeError):
        return None

    sentiment = map_rating_to_sentiment(rating)
    if sentiment is None:
        return None

    return dspy.Example(
        **{
            DSPyReviewSchema.TEXT: text,
            DSPyReviewSchema.SENTIMENT: sentiment,
        }
    ).with_inputs()


def load_b2w_reviews(limit: int | None = None) -> list[Example]:
    """
    Carrega o dataset B2W Reviews e retorna exemplos DSPy.
    """

    print("Carregando dataset B2W Reviews...")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} não encontrado. Execute download primeiro.")


    examples = []

    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            example = b2w_review(row)
            if example is None:
                continue
            
            examples.append(example)

            if limit is not None and len(examples) >= limit:
                break
            
    print(f"Dataset carregado com {len(examples)} exemplos.")
    
    return examples
