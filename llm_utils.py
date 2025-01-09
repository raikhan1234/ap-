from ollama import Ollama

def query_llm(question: str) -> str:
    model = Ollama("llama2")
    return model.generate(question)
