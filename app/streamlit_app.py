import streamlit as st
from src.models.predict import predict_next_day
from src.llm.prompts import explain_stock_prediction

st.title("ML Stock Portfolio Manager - Starter")

st.write("Upload your stock CSV (OHLC)")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Predictions:")
    predict_next_day("temp.csv")

    st.write("LLM Explanation Example:")
    st.write(explain_stock_prediction("Tesla", 1, 0.62))