import dspy
import os
from domain.evaluation.sentiment_eval import run_evaluation
from domain.evaluation.sentiment_opt import run_optimization


def main():
    llm_local_mode = os.getenv("DSPY_AI_LOCAL_MODE", "false").lower()
    if llm_local_mode == "true":
        llm = dspy.LM(
            model="ollama/glm4:9b-chat-q3_K_M",
            chat=True,
            max_tokens=256,
            local_mode=True
        )
        print("Usando modelo local (Ollama GLM4).")
    else:
        llm = dspy.LM(
            model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            chat=True,
            max_tokens=256
        )
        print("Usando modelo remoto (Liquid LFM 2.5).")

    dspy.settings.configure(lm=llm)

    run_evaluation()
    run_optimization()


if __name__ == "__main__":
    main()
