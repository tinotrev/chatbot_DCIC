
# app/rag_agent.py
import numpy as np
import pandas as pd
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login

# Autenticación (mejor mover esto a una función para mayor seguridad)
login(token="***********")

class RAGAgent:
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Inicializar los modelos al crear la instancia
        self.tokenizer, self.model, self.semantic_similarity_model = self._load_models()
        
        # Cargar los datos necesarios para el agente RAG
        self.df, self.listaPreguntas, self.preguntasEmbeddings = self._load_data(self.semantic_similarity_model)
        
    def _load_models(self):
        """Carga y devuelve el tokenizer, modelo y modelo de similaridad semántica"""
        try:
            print("Cargando tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
            
            print("Cargando modelo Gemma...")
            # Verificar si CUDA está disponible
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Usando dispositivo: {device}")
            
            # Cargar modelo con configuración optimizada
            model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2b-it",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None
            )
            
            # Si no tienes GPU, cargar en CPU
            if device == "cpu":
                model = model.to("cpu")
            
            print("Cargando modelo de similitud semántica...")
            semantic_similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            print("¡Todos los modelos cargados correctamente!")
            return tokenizer, model, semantic_similarity_model
            
        except Exception as e:
            print(f"Error al cargar modelos: {e}")
            raise e
    
    def _load_data(self, semanticSimilarityModel):
        """Carga los datos necesarios para el agente RAG"""
        try:
            # Ajustar la ruta del archivo CSV
            csv_path = "../dataset_tutorias.csv"
            if not os.path.exists(csv_path):
                # Intentar con rutas alternativas
                alternative_paths = [
                    "./dataset_tutorias.csv",
                    "dataset_tutorias.csv",
                    "../app/dataset_tutorias.csv"
                ]
                for path in alternative_paths:
                    if os.path.exists(path):
                        csv_path = path
                        break
                else:
                    raise FileNotFoundError(f"No se encontró el archivo dataset_tutorias.csv en las rutas: {[csv_path] + alternative_paths}")
            
            print(f"Cargando datos desde: {csv_path}")
            df = pd.read_csv(csv_path)
            listaPreguntas = df.Pregunta.tolist()
            
            print("Generando embeddings de preguntas...")
            preguntasEmbeddings = semanticSimilarityModel.encode(listaPreguntas, show_progress_bar=False)
            
            print(f"Cargadas {len(listaPreguntas)} preguntas")
            return df, listaPreguntas, preguntasEmbeddings
            
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            raise e
    
    def generarTemplatePregunta(self, contexto, pregunta):
        template = f"""Eres un asistente virtual altamente competente en el área administrativa. Se te proporciona el siguiente contexto:

{contexto}

Con base en esta información, responde a la siguiente pregunta de manera clara, concisa y sin agregar información que no corresponda:

Pregunta: {pregunta}
Respuesta:"""
        return template
    
    def generarTemplateRespuestaBorrador(self, contexto, pregunta):
        template = f"""Eres un asistente virtual altamente competente en el área administrativa. Se te proporciona el siguiente contexto:

{contexto}

Con base en el contexto proporcionado, genera un template de respuesta que pueda ser utilizado para responder preguntas similares sobre este tema sin agregar información que no corresponda y en particular la siguiente:

Pregunta: {pregunta}
Respuesta:"""
        return template
    
    def generarContexto(self, preguntaUsuario):
        try:
            preguntaUsuarioEmbedding = self.semantic_similarity_model.encode(preguntaUsuario, show_progress_bar=False)
            hits = util.semantic_search(preguntaUsuarioEmbedding, self.preguntasEmbeddings, top_k=3, score_function=util.cos_sim)
            hits = hits[0]
            
            contexto = ''
            for hit in hits:
                preguntaSimilar = self.listaPreguntas[hit['corpus_id']]
                respuesta = self.df.Respuesta[self.df.Pregunta == preguntaSimilar].iloc[0]
                contexto += f'Pregunta: {preguntaSimilar}\nRespuesta: {respuesta}\n\n'
            
            return contexto
        except Exception as e:
            print(f"Error al generar contexto: {e}")
            return "No se pudo generar contexto para la pregunta."
    
    def generarPromptPregunta(self, preguntaUsuario):
        contexto = self.generarContexto(preguntaUsuario)
        chat = [{"role": "user", "content": self.generarTemplatePregunta(contexto, preguntaUsuario)}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def generarPromptbBorradorRespuesta(self, preguntaUsuario):
        contexto = self.generarContexto(preguntaUsuario)
        chat = [{"role": "user", "content": self.generarTemplateRespuestaBorrador(contexto, preguntaUsuario)}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    def preguntar(self, preguntaUsuario):
        try:
            prompt = self.generarPromptPregunta(preguntaUsuario)
            inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            
            # Asegurar que los inputs estén en el mismo dispositivo que el modelo
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():  # Optimización de memoria
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=256,
                    num_beams=4,
                    penalty_alpha=0.6,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraer solo la respuesta del modelo
            if '<start_of_turn>model' in response:
                response = response.split('<start_of_turn>model')[1]
            if '<eos>' in response:
                response = response.split('<eos>')[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"Error al generar respuesta: {e}")
            return "Lo siento, no pude generar una respuesta en este momento."
    
    def respuestaBorrador(self, preguntaUsuario):
        try:
            prompt = self.generarPromptbBorradorRespuesta(preguntaUsuario)
            inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=512,
                    num_beams=3,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if '<start_of_turn>model' in response:
                response = response.split('<start_of_turn>model')[1]
            if '<eos>' in response:
                response = response.split('<eos>')[0]
                
            return response.strip()
            
        except Exception as e:
            print(f"Error al generar borrador: {e}")
            return "Lo siento, no pude generar un borrador en este momento."
    
    def generate_response(self, question: str) -> str:
        """Método principal para generar respuestas"""
        try:
            respuesta = self.preguntar(question)
            return respuesta
        except Exception as e:
            print(f"Error en generate_response: {e}")
            return "Lo siento, ocurrió un error al procesar tu pregunta."