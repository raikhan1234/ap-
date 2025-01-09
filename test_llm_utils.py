from src.llm_utils import query_llm

def test_query_llm():
    response = query_llm("What is the capital of France?")
    assert "Paris" in response  # Example test
