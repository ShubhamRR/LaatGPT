import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.header("Laat GPT")
st.write("A personal and secure friend of yours")

with st.sidebar:
    st.title("Laat GPT")
    file = st.file_uploader("Upload your file", type=["pdf"])

# Extract content from the uploaded file
if file is not None:
    # Read the file content
    with pdfplumber.open(file) as pdf:
        text =""
        for page in pdf.pages:
            text += page.extract_text()

    st.write("Extracted Text:", text)

# split  text into chunks
    text_splitter =  RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap =200
    )
    chunks = text_splitter.split_text(text)
    st.write("Text Chunks:", chunks)