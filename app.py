from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import connect_to_db, IntelligentRAGSystem, classifier

app = FastAPI(title="Jordan RAG API")

# لو عندك دومين للفرونت اند بدّله بدل "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # بالأنتاج حدد نطاقك بدل *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    # نفتح اتصال Weaviate عند تشغيل السيرفر
    global client
    client, ready = connect_to_db()
    print("✅ Weaviate connected:", ready)

@app.on_event("shutdown")
def shutdown_event():
    # نغلق الاتصال عند إطفاء السيرفر
    try:
        client.close()
    except:
        pass

@app.post("/ask")
def ask_question(body: AskRequest):
    answer = IntelligentRAGSystem(body.query, client, classifier)
    return {"question": body.query, "answer": answer}
