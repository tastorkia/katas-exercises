# rag/evals.py
import json
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from agent import agent as rag_agent  # Import your agent from rag/agent.py


# 1. Define Judge Output Schema
class EvalScore(BaseModel):
    is_faithful: bool = Field(
        description="True if final_answer relies ONLY on the retrieved source_snippet without inventing facts."
    )
    is_relevant: bool = Field(
        description="True if the answer directly answers the question asked."
    )
    score: int = Field(
        description="Rating from 1 (poor) to 5 (excellent) based on answer quality."
    )
    reasoning: str = Field(description="Explanation for the score.")


# 2. Create the LLM Judge Agent
judge_agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    output_type=EvalScore,
    system_prompt=(
        "You are an impartial AI judge evaluating a RAG system. "
        "Analyze the user query, the retrieved source snippet, and the agent's final answer. "
        "Determine if the answer is faithful to the source and directly relevant to the question."
    ),
)

# 3. Test Cases for Evaluation
TEST_DATASET = [
    {
        "query": "What is the primary benefit of performing The Hundred?",
        "expected_topic": "warm-up and circulation",
    },
    {
        "query": "How should I breathe during The Roll Up?",
        "expected_topic": "breathing",
    },
]


def run_evals():
    print("🧪 Running RAG Agent Evaluations...\n" + "=" * 50)

    for i, test in enumerate(TEST_DATASET, 1):
        query = test["query"]
        print(f"\n[Test {i}] Query: {query}")

        # Run your RAG agent
        rag_result = rag_agent.run_sync(query)
        response = rag_result.output

        # Evaluate using the LLM Judge
        judge_prompt = f"""
        USER QUESTION: {query}
        RETRIEVED SNIPPET: {response.source_snippet}
        GENERATED ANSWER: {response.final_answer}
        """

        eval_result = judge_agent.run_sync(judge_prompt)
        score_data: EvalScore = eval_result.output

        # Print Evaluation Output
        print(f"  📌 Faithful to Source? {score_data.is_faithful}")
        print(f"  🎯 Relevant to Query? {score_data.is_relevant}")
        print(f"  ⭐ Score: {score_data.score}/5")
        print(f"  💬 Judge Reasoning: {score_data.reasoning}\n" + "-" * 50)


if __name__ == "__main__":
    run_evals()