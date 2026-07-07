import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai  import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", ""),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)


class OpenRouterEmbeddings(Embeddings):

    def embed_documents(self, texts):
        try:
            # st.write("Generating embeddings for the provided texts...", texts)
            response = client.embeddings.create(
                model=os.getenv(
                    "OPENROUTER_EMBEDDING_MODEL",
                    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                ),
                input=texts,
                encoding_format="float",
            )
            if not response:
                raise Exception(f"No embeddings returned: {response}")

            # st.write(f"response: {response}")
            return [item.embedding for item in response.data]

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None

    def embed_query(self, text):
        try:
            response = client.embeddings.create(
                model=os.getenv(
                    "OPENROUTER_EMBEDDING_MODEL",
                    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                ),
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for query: e{e}")
            return None



st.header("Laat GPT")
st.write("A personal friend of yours")

with st.sidebar:
    st.title("Laat GPT")
    file = st.file_uploader("Upload your file", type=["pdf"])

# Extract content from the uploaded file
if file is not None:
    # Read the file content
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # split  text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    # generating embeddings and vector storing
    embeddings = OpenRouterEmbeddings()
    # st.write("Embeddings are ready for vector storage.", embeddings)
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    # user question input
    user_question = st.text_input("How may I help yo today?")

    # generate response based on user question

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # Define llm
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_LLM_MODEL", "nvidia/nemotron-nano-9b-v2:free"),
        temperature=0.7,
        max_tokens=1000,
        openai_api_key=os.getenv("OPENROUTER_API_KEY")
        or st.secrets.get("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    # prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. answering the question based on the context provided.\n\n"
                "Guidelines:\n"
                "1. Provide complete, well-explained answers. using the context provided\n"
                "2. Include relevant details, numbers, and examples from the context tp support your answer.\n"
                "3. If the context mentions related information, include it in your answer.\n"
                "4. Only use the context provided to answer the question. DO not use any external information or make assumptions.\n"
                "5. If the context does not contain enough information to answer the question, respond with 'I don't know' or 'The context does not provide enough information to answer the question.'\n"
                "6. Summarize the answer in a concise manner, highlighting the key points and main takeaways.\n"
                "7. If the information is not there in the context, do not make up an answer. Instead, respond with 'I don't know' or 'The context does not provide enough information to answer the question.'\n"
                "Context:\n{context}\n\n",
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)
