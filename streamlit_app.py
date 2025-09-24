import os
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import HuggingFaceHub

st.title("CSV Q&A — Company Funding Data")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df.head())

    # ✅ Set Hugging Face API key from Streamlit secrets
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_API_KEY"]

    # ✅ Hugging Face model (no huggingfacehub_api_token arg anymore)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0, "max_length": 512}
    )

    # ✅ Create CSV Agent
    agent = create_csv_agent(
        llm,
        uploaded_file,
        verbose=True
    )

    # Chatbot input
    query = st.text_input("Ask a question about your CSV data:")

    if query:
        with st.spinner("Thinking..."):
            try:
                answer = agent.run(query)
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
