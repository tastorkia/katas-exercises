import os
import chromadb
from dotenv import load_dotenv
from pydantic import BaseModel, Field  # 1. Added Pydantic import
from pydantic_ai import Agent

os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

# Load environment variables (API keys)
load_dotenv()


# 2. Define Structured Output Schema required by assignment
class PilatesRAGResponse(BaseModel):
    final_answer: str = Field(
        description="The grounded answer to the user's question."
    )
    source_snippet: str = Field(
        description="The exact snippet or text quote retrieved from the knowledge base that was used to answer."
    )


# Initialize ChromaDB persistent client
base_dir = os.path.dirname(os.path.abspath(__file__))
db_folder_path = os.path.join(base_dir, "chroma_db")
client = chromadb.PersistentClient(path=db_folder_path)
collection = client.get_collection(name="pilates_kb")

# Initialize Pydantic AI Agent with Structured Result Type
# Initialize Pydantic AI Agent with Structured Output
agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    output_type=PilatesRAGResponse,  # <-- Change result_type to output_type
    system_prompt=(
        "You are an expert Pilates instructor and assistant. "
        "Always use the 'search_pilates_knowledge_base' tool to look up details about "
        "exercises, their importance, and execution steps before answering user questions. "
        "Rely on the retrieved facts from the knowledge base."
    ),
)


@agent.tool_plain
def search_pilates_knowledge_base(query: str) -> str:
    """Searches the classical Pilates vector store for exercise rules, importance, and execution steps.

    Args:
        query: The topic or exercise name to search for (e.g., 'The Teaser', 'breathing', 'spine articulation').
    """
    results = collection.query(query_texts=[query], n_results=3)

    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No relevant information found in the Pilates knowledge base."

    return "\n\n---\n\n".join(docs)


def main():
    print("🧘 Classical Pilates RAG Agent Ready!")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        user_query = input("Ask a question about Pilates: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break

        if not user_query.strip():
            continue

        print("\nThinking...")
        result = agent.run_sync(user_query)

        # 4. Access structured fields from response
        response: PilatesRAGResponse = result.output

        print(f"\n🤖 Answer:\n{response.final_answer}\n")
        print(f"📌 Source Snippet:\n{response.source_snippet}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()