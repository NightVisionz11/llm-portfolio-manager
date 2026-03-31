def explain_stock_prediction(stock_name: str, prediction: int, probability: float):
    """
    Returns a simple prompt for an LLM to explain a stock prediction.
    """
    action = "up" if prediction == 1 else "down"
    return (
        f"The model predicts that {stock_name} stock will go {action} tomorrow "
        f"with a probability of {probability:.2f}. Explain this in simple terms for a beginner investor."
    )

if __name__ == "__main__":
    print(explain_stock_prediction("Tesla", 1, 0.62))