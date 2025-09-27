import os
import streamlit as st
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import HuggingFaceHub

import tempfile

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df.head())

    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    # Hugging Face model
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # try base first for speed
        task="text2text-generation",
        model_kwargs={"temperature": 0.0, "max_length": 512}
    )

    # Create CSV agent (now with correct file path)
    agent = create_csv_agent(
        llm,
        tmp_file_path,   # 👈 pass file path, not uploaded_file
        verbose=True
    )

    # Chatbot part
    query = st.text_input("Ask a question about your CSV data:")

    if query:
        with st.spinner("Thinking..."):
            answer = agent.run(query)
            st.write(answer)
