# src/domain/evaluation/logger.py

from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "dspy_results.txt"


def log_result(
    phase: str,
    metric_name: str,
    metric_value: float,
    num_examples: int,
    model_name: str,
    notes: str = ""
):
    timestamp = datetime.utcnow().isoformat()

    line = (
        f"{timestamp};"
        f"{phase};"
        f"{metric_name};"
        f"{metric_value:.4f};"
        f"{num_examples};"
        f"{model_name};"
        f"{notes}\n"
    )
    print("Logging result:", line.strip())
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(line)
