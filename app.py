from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pipeline import connect_to_db, IntelligentRAGSystem, classifier

app = FastAPI(title="Jordan RAG API")

# موديل الطلب
class AskRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    global client
    client, ready = connect_to_db()
    print("✅ Connected:", ready)

@app.on_event("shutdown")
def shutdown_event():
    try:
        client.close()
    except:
        pass

@app.post("/ask")
def ask_question(body: AskRequest):
    answer = IntelligentRAGSystem(body.query, client, classifier)
    return {"question": body.query, "answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
