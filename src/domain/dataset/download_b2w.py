import requests
from pathlib import Path


B2W_REVIEWS_URL = (
    "https://raw.githubusercontent.com/b2w-digital/b2w-reviews01/main/B2W-Reviews01.csv"
)


def download_b2w_reviews(force: bool = False) -> Path:
    """
    Faz o download do dataset B2W Reviews e salva localmente.
    Retorna o caminho do arquivo.
    """

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / "b2w_reviews.csv"

    if output_file.exists() and not force:
        print("Dataset B2W Reviews já existe. Pulando download.")
        return output_file

    print("Baixando dataset B2W Reviews...")

    response = requests.get(B2W_REVIEWS_URL, timeout=60)
    response.raise_for_status()

    output_file.write_bytes(response.content)

    print(f"Dataset salvo em: {output_file}")
    
    return output_file