# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import RAGAgent  # Cambiado de agent a rag_agent

app = FastAPI()

# Crear una instancia global del agente
try:
    rag_agent = RAGAgent()
except Exception as e:
    print(f"Error al inicializar RAGAgent: {e}")
    rag_agent = None

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(query: Query):
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="El agente no está disponible")
    
    response = rag_agent.generate_response(query.message)
    return {"response": response}

@app.get("/chat")
def test_get():
    return {"message": "Chatbot API está funcionando. Usa POST para enviar mensajes."}