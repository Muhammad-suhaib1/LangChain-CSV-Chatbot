import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import HuggingFaceHub

st.title("CSV Q&A — Company Funding Data")

# CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df.head())

    # Hugging Face model
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0, "max_length": 512},
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )

    # Creating CSV agent
    agent = create_csv_agent(
        llm,
        uploaded_file,
        verbose=True
    )

    # The Chatbot Stuff
    query = st.text_input("Ask a question about your CSV data:")

    if query:
        with st.spinner("Thinking..."):
            answer = agent.run(query)
            st.write(answer)
