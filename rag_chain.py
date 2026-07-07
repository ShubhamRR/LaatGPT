from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant. answering the question based on the context provided.\n\n"
    "Guidelines:\n"
    "1. Provide complete, well-explained answers. using the context provided\n"
    "2. Include relevant details, numbers, and examples from the context tp support your answer.\n"
    "3. If the context mentions related information, include it in your answer.\n"
    "4. Only use the context provided to answer the question. DO not use any external information or make assumptions.\n"
    "5. If the context does not contain enough information to answer the question, respond with 'I don't know' or 'The context does not provide enough information to answer the question.'\n"
    "6. Summarize the answer in a concise manner, highlighting the key points and main takeaways.\n"
    "7. If the information is not there in the context, do not make up an answer. Instead, respond with 'I don't know' or 'The context does not provide enough information to answer the question.'\n"
    "Context:\n{context}\n\n"
)


class RAGchain:
    def __init__(self, retriever):
        self.llm = ChatOpenAI(
            model=settings.OPENROUTER_LLM_MODEL,
            temperature=0.7,
            max_tokens=1000,
            openai_api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}")
            ]
        )

        self.chain = (
            {"context":retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question):
        return self.chain.invoke(question)
    
    def stream(self, question):
        return self.chain.stream(question)
    