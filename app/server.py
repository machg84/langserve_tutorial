from langchain_openai import OpenAI
from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_host = os.getenv("QDRANT_HOST")

embeddings = OpenAIEmbeddings()

llm = OpenAI(model='gpt-3.5-turbo-instruct',
             temperature=0.5)

qdrant = QdrantVectorStore.from_existing_collection(
    api_key=qdrant_api_key,
    collection_name="sample collection",
    url=qdrant_host ,
    embedding=embeddings
)

retriever = qdrant.as_retriever()

prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

# Instructions
- Answer in spanish

Question: {question}

Context: {context}""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="NCG 454 App",
)

add_routes(
    app,
    rag_chain,
    path="/openai"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
