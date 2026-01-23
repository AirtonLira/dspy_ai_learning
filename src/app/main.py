import dspy
import os
from dotenv import load_dotenv
from domain.evaluation.sentiment_eval import run_evaluation
from domain.evaluation.sentiment_opt import run_optimization
from utils.config import setup_llm

load_dotenv()


def main():
  
    setup_llm()

    #run_evaluation()
    run_optimization()


if __name__ == "__main__":
    main()
