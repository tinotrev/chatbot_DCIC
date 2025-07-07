# app/rag_agent.py
import numpy as np
import pandas as pd
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

class RAGAgent:
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Inicializar los modelos al crear la instancia
        self.tokenizer, self.model, self.semantic_similarity_model = self._load_models()
        
        # Cargar los datos necesarios para el agente RAG
        self.df, self.listaPreguntas, self.preguntasEmbeddings = self._load_data(self.semantic_similarity_model)
        
    def _load_models(self):
        """Carga y devuelve el tokenizer, modelo y modelo de similaridad semántica"""
        tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/gemma/transformers/2b-it/3")
        model = AutoModelForCausalLM.from_pretrained("/kaggle/input/gemma/transformers/2b-it/3", device_map="auto")
        semantic_similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return tokenizer, model, semantic_similarity_model
    
    def _load_data(self, semanticSimilarityModel):
        """Carga los datos necesarios para el agente RAG"""
        df = pd.read_csv("../Dataset Bot Tutorias.csv")
        listaPreguntas = df.Pregunta.tolist()
        preguntasEmbeddings = semanticSimilarityModel.encode(listaPreguntas, show_progress_bar=False)
        return df, listaPreguntas, preguntasEmbeddings
    
    def generarTemplatePregunta(self, contexto, pregunta):
        template = f"""Eres un asistente virtual altamente competente en el área administrativa.Se te proporciona el siguiente
        contexto:{contexto}
        Con base en esta información, responde a la siguiente pregunta de manera clara, concisa y sin agregar informacion que no corresponda
        pregunta:{pregunta}
        Respuesta:"""
        return template
    
    def generarTemplateRespuestaBorrador(self, contexto, pregunta):
        template = f"""Eres un asistente virtual altamente competente en el área administrativa.Se te proporciona el siguiente
        contexto: {contexto}
        Con base en el contexto proporcionado, genera un template de respuesta que pueda ser utilizado para responder preguntas similares sobre este tema sin agregar informacion que no corresponda y en particular la siguiente.
        Pregunta: {pregunta}
        Respuesta:"""
        return template
    
    def generarContexto(self, preguntaUsuario):
        preguntaUsuarioEmbedding = self.semantic_similarity_model.encode(preguntaUsuario,show_progress_bar=False)
        hits = util.semantic_search(preguntaUsuarioEmbedding, self.preguntasEmbeddings, top_k = 3, score_function = util.cos_sim)
        hits = hits[0]
        contexto = ''
        for hit in hits:
            preguntaSimilar = self.listaPreguntas[hit['corpus_id']]
            respuesta = self.df.Respuesta[self.df.Pregunta == preguntaSimilar].iloc[0]
            contexto+= f'{preguntaSimilar}\n{respuesta}\n'
        return contexto
    
    def generarPromptPregunta(self, preguntaUsuario):
        contexto = self.generarContexto(preguntaUsuario)
        chat = [{ "role": "user", "content": self.generarTemplatePregunta(contexto,preguntaUsuario)}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def generarPromptbBorradorRespuesta(self, preguntaUsuario):
        contexto = self.generarContexto(preguntaUsuario)
        chat = [{ "role": "user", "content": self.generarTemplateRespuestaBorrador(contexto,preguntaUsuario)}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    def preguntar(self, preguntaUsuario):
        inputs = self.tokenizer.encode(self.generarPromptPregunta(preguntaUsuario), add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=256,num_beams=4,penalty_alpha=0.6)
        return self.tokenizer.decode(outputs[0]).split('<start_of_turn>model')[1].split('<eos>')[0]
    
    def respuestaBorrador(self, preguntaUsuario):
        inputs = self.tokenizer.encode(self.generarPromptbBorradorRespuesta(preguntaUsuario), add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=512,num_beams=3)
        return self.tokenizer.decode(outputs[0]).split('<start_of_turn>model')[1].split('<eos>')[0]
    
    def generate_response(self, question: str) -> str:
        respuesta = self.preguntar(question)
        return respuesta
        