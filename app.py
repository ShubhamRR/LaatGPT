import streamlit as st

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGchain
from chat_ui import inject_chat_ui_css, render_chat_message
# st.header("Laat GPT")
# st.write("A personal friend of yours")
inject_chat_ui_css()
st.set_page_config(page_title="Laat GPT", layout="centered")
st.title("Laat GPT")

with st.sidebar:
    st.header("Laat GPT")
    file = st.file_uploader("Upload your file", type=["pdf"])
    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.pop("rag_chain", None)
        st.session_state.pop("processed_file", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if file is not None:
    # Only rebuild embeddings/vector store when a NEW file is uploaded
    if st.session_state.get("processed_file") != file.name:
        with st.spinner("Reading and indexing document..."):
            processor = DocumentProcessor()
            chunks = processor.process(file)

            vector_store_manager = VectorStoreManager()
            vector_store_manager.build(chunks)
            retriever = vector_store_manager.as_retriever(search_type="mmr", k=3)

            st.session_state.rag_chain = RAGchain(retriever)
            st.session_state.processed_file = file.name
            st.session_state.messages = []

    # Replay chat history on every rerun
    for message in st.session_state.messages:
        render_chat_message(message["role"], message["content"])

    # rag_chain = RAGchain(retriever)
    
    user_question = st.chat_input("How may I help you today?")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        render_chat_message("user", user_question)

        placeholder = st.empty()
        full_response =""
        for chunk in st.session_state.rag_chain.stream(user_question):
            full_response += chunk
            with placeholder.container():
                render_chat_message("assistant", full_response)

        # with st.chat_message("assistant"):
        #     with st.spinner("Generating response..."):
        #         response = st.write_stream(st.session_state.rag_chain.stream(user_question))

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Upload a PDF from the sidebar to start chatting.")
        
        
        # response = rag_chain.ask(user_question)
        # st.write(response)
