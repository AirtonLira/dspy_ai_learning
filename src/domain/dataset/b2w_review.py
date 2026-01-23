from dspy import Example
import csv
from pathlib import Path


def load_b2w_reviews(limit: int | None = None) -> list[Example]:
    """
    Carrega o dataset B2W Reviews e retorna exemplos DSPy.
    """

    print("Carregando dataset B2W Reviews...")
    dataset_path = Path(__file__).parent / "data" / "b2w_reviews.csv"

    examples = []

    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            sentiment = "positivo" if int(row["overall_rating"]) >= 4 else "negativo"

            example = (
                Example(
                    text=row["review_text"],
                    sentiment=sentiment
                )
                .with_inputs("text")
            )

            examples.append(example)

            if limit and i + 1 >= limit:
                break
    print(f"Dataset carregado com {len(examples)} exemplos.")
    
    return examples
