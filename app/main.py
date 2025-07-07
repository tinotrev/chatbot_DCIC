# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent import RAGAgent  # Importa la clase, no una función

app = FastAPI()

# Crear una instancia global del agente (se carga una vez al iniciar)
rag_agent = RAGAgent()

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(query: Query):
    response = rag_agent.generate_response(query.message)  # Usa la pregunta del usuario
    return {"response": response}