# src/llm/explainer.py
import anthropic
from src.llm.prompts import explain_stock_prediction

def get_llm_explanation(stock_name: str, prediction: int, probability: float) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = explain_stock_prediction(stock_name, prediction, probability)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text