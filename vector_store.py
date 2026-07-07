from langchain_community.vectorstores import FAISS
from embeddings import OpenRouterEmbeddings

class VectorStoreManager:
    def __init__(self):
        self.embeddings= OpenRouterEmbeddings()
        self.vector_store = None

    def build(self, chunks):
        self.vector_store= FAISS.from_texts(texts= chunks, embedding=self.embeddings)
        return self.vector_store
    
    def as_retriever(self, search_type="mmr", k=3):
        if self.vector_store is None:
            raise ValueError("Vector store has not been built yet. call build() first.")
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k":k})
    
    