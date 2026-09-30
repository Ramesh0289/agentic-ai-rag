from src.app import ChatRequest


def test_chat_request():
    request = ChatRequest(query="What is Agentic AI?")
    assert request.query == "What is Agentic AI?"
