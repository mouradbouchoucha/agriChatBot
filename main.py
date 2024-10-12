from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = Ollama(model="mistral")

@app.get("/")
def root():
    return {"message": "Welcome to the AgriChatBot API"}

@app.post("/query/")
def query_rag(request: QueryRequest):
    query_text = request.query
    try:
        results = db.similarity_search_with_score(query_text, k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    try:
        response_text = model.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    return {"response": response_text, "sources": sources}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
