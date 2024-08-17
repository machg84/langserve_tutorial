from langchain_openai import OpenAI
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
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
app_api_key = os.getenv("APP_API_KEY")

embeddings = OpenAIEmbeddings()

llm = OpenAI(model='gpt-3.5-turbo-instruct',
             temperature=0.5)

qdrant = QdrantVectorStore.from_existing_collection(
    api_key=qdrant_api_key,
    collection_name="CMF",
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
    description="CMF App",
)

# Definir la dependencia para la API key
api_key_header = APIKeyHeader(name="X-API-KEY")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != app_api_key:
        raise HTTPException(status_code=403, detail="Could not validate API key")

# Proteger la ruta con la dependencia
@app.get("/openai", dependencies=[Depends(verify_api_key)])
async def openai_endpoint():
    return {"message": "API key is valid"}

add_routes(
    app,
    rag_chain,
    path="/openai"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
