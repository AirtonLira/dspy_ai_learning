import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# from domain.dataset.download_b2w import download_b2w_reviews


def main():
    # download_b2w_reviews()
    print("setup_dataset.py - Dataset já está em src/domain/dataset/data/b2w_reviews.csv")


if __name__ == "__main__":
    main()
