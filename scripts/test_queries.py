from src.graph import RAGGraph

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What real-world industry use cases are discussed in the eBook?",
    "Who won the 2022 FIFA World Cup?",
]


def main():
    graph = RAGGraph()
    for number, query in enumerate(QUERIES, start=1):
        print("\n" + "=" * 70)
        print(f"TEST {number}: {query}")
        result = graph.ask(query)
        print("ANSWER:", result["final_answer"])
        print("SCORE:", result["confidence_score"])
        print("RETRIEVED CHUNKS:")
        for chunk in result["retrieved_context_chunks"]:
            print("-", chunk[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
