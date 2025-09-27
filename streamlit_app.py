import os
import streamlit as st
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import HuggingFaceHub

st.title("CSV Q&A — Company Funding Data")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df.head())

    # Load Hugging Face API key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_API_KEY"]

    # Hugging Face model
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0, "max_length": 512}
    )

    # Create CSV agent
    agent = create_csv_agent(
        llm,
        uploaded_file,
        verbose=True
    )

    # Chatbot interface
    query = st.text_input("Ask a question about your CSV data:")

    if query:
        with st.spinner("Thinking..."):
            try:
                answer = agent.run(query)
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
