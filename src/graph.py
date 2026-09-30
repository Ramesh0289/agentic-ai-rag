from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.config import get_settings
from src.vector_store import VectorStore


class AgentState(TypedDict, total=False):
    question: str
    context: list[dict]
    answer: str
    score: float


class RAGGraph:
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.llm = self._create_llm()
        self.graph = self._build_graph()

    def _create_llm(self):
        if self.settings.llm_provider.lower() == "ollama":
            import ollama
            return ollama

        if self.settings.llm_provider.lower() == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI mode.")
            from openai import OpenAI
            return OpenAI(api_key=self.settings.openai_api_key)

        raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai'.")

    def retrieve_node(self, state: AgentState):
        matches = self.vector_store.search(state["question"], self.settings.top_k)
        matches = [
            item for item in matches
            if item["score"] >= self.settings.relevance_threshold
        ]
        return {"context": matches}

    def generate_node(self, state: AgentState):
        context = state.get("context", [])
        refusal = "I cannot answer this question based on the provided document."

        if not context:
            return {"answer": refusal, "score": 0.0}

        context_text = "\n\n".join(
            f"[Page {item['page']}] {item['text']}" for item in context
        )

        prompt = f"""
You are a document question-answering assistant.

Answer the question ONLY using the context from the Agentic AI eBook below.
Do not use outside knowledge. Do not guess.
If the answer is not present, say exactly:
{refusal}
Keep the answer simple and clear.

Context:
{context_text}

Question:
{state['question']}
""".strip()

        if self.settings.llm_provider.lower() == "ollama":
            response = self.llm.chat(
                model=self.settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response["message"]["content"].strip()
        else:
            response = self.llm.responses.create(
                model=self.settings.openai_chat_model,
                input=prompt,
            )
            answer = response.output_text.strip()

        score = max(0.0, min(1.0, max(item["score"] for item in context)))
        if refusal.lower() in answer.lower():
            score = min(score, 0.20)

        return {"answer": answer, "score": round(score, 2)}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def ask(self, question: str) -> dict:
        result = self.graph.invoke({"question": question})
        return {
            "final_answer": result["answer"],
            "retrieved_context_chunks": [
                item["text"] for item in result.get("context", [])
            ],
            "confidence_score": result.get("score", 0.0),
        }
