from dotenv import load_dotenv
from domain.evaluation.sentiment_opt_fewshot import run_optimization
from domain.evaluation.sentiment_opt_mipro_v2 import SentimentMiproManager
from utils.config import setup_llm

load_dotenv()


def main():
  
    setup_llm()

    #run_evaluation()
    #run_optimization()
    manager = SentimentMiproManager()
    manager.run_mipro_optimization(num_candidates=3)
    manager.save_checkpoint("sentiment_mipro_final.json")
    


if __name__ == "__main__":
    main()
