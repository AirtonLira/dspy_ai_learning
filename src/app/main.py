import dspy
from domain.evaluation.sentiment_eval import run_evaluation


def main():
    llm = dspy.LM(
        model="ollama/glm4:9b-chat-q5_0",
        chat=True,
        max_tokens=256
    )

    dspy.settings.configure(lm=llm)

    run_evaluation()


if __name__ == "__main__":
    main()
