from openai import OpenAI
from langchain_core.embeddings import Embeddings

from config import settings

class OpenRouterEmbeddings(Embeddings):
    def __init__(self):
        self.client=OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL
        )
        self.model=settings.OPENROUTER_EMBEDDING_MODEL

    def embed_documents(self, texts):
        try:
            # st.write("Generating embeddings for the provided texts...", texts)
            response = self.client.embeddings.create(
                model=self.model,
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
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for query: e{e}")
            return None