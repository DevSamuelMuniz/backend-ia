import uuid

#env
import os
from dotenv import load_dotenv

#FASTAPI WEBSOCKET
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# modelo
from modelo_predictor import SVCService
from predict import SVCInput
import openai

# agentes
from question_agent import QuestionAgent
from validador_agent import ValidadorAgent
from lista_agent import ListaAgent
from diag_agent import DiagAgent

# firebase
import firebase_admin
from firebase_admin import credentials, firestore
import os
from firebase_admin import credentials, initialize_app
from dotenv import load_dotenv

# Inicializar Firebase (garantir que sempre exista o db)
cred = credentials.Certificate(os.getenv("FIREBASE_KEY_PATH"))
initialize_app(cred)

try:
    firebase_admin.initialize_app(cred)
except ValueError:
    # já estava inicializado
    pass

db = firestore.client()
#

load_dotenv()

# API Key OpenAI
openai.api_key = os.getenv("APIKEY_OPENAI")
if not openai.api_key:
    raise RuntimeError("A variável OPENAI_API_KEY não está definida!")

app = FastAPI()
svc = SVCService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variável de controle global
concluiu_o_questionario_e_foi_validado = False
modo_diagnostico_ativo = False

# Função para predição com o modelo
async def predict_cancer(payload: SVCInput):
    resposta_ml = svc.predict(payload)

    if resposta_ml["prediction"] == 1:
        mensagem = "com câncer"
    else:
        mensagem = "sem câncer"

    shap_values_classe_predita = resposta_ml["shap_values"]

    return {
        "resultado": mensagem,
        "probabilidade": resposta_ml["probability"],
        "shap_values": shap_values_classe_predita
    }

async def salvar_no_firestore(resultado: dict, lista_json: list):
    try:
        doc_ref = db.collection("resultados_predicao").document()
        doc_ref.set({
            "resultado": resultado["resultado"],
            "probabilidade": resultado["probabilidade"],
            "shap_values": resultado["shap_values"],
            "features": lista_json
        })
        print("✅ Resultado da predição salvo com sucesso no Firestore.")
    except Exception as e:
        print(f"❌ Erro ao salvar resultado no Firestore: {e}")


@app.websocket("/agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Conexão Aceita")

    global concluiu_o_questionario_e_foi_validado
    global modo_diagnostico_ativo

    question_agent = QuestionAgent(websocket, openai=openai, isAsync=False)
    validador_agent = ValidadorAgent(websocket, openai=openai, isAsync=False)
    json_agent = ListaAgent(websocket, openai=openai, isAsync=False)
    diag_agent = DiagAgent(websocket, openai=openai, isAsync=False)

    conversation_history = []

    try:
        while True:
            id = str(uuid.uuid4())
            question = await websocket.receive_text()
            print(f"📩 Recebido: {question}")

            conversation_history.append({"role": "user", "content": question})

            response = None
            if not modo_diagnostico_ativo:

                if not concluiu_o_questionario_e_foi_validado:
                    response = await question_agent.handle(id, question, contexto=conversation_history[:])

                if response is not None:
                    conversation_history.append({"role": "assistant", "content": response})
                    concluiu_o_questionario_e_foi_validado = await validador_agent.handle(id, question, contexto=conversation_history[:])

                print(f"✅ Validação concluída: {concluiu_o_questionario_e_foi_validado}")
                
                if concluiu_o_questionario_e_foi_validado:
                    lista_json = await json_agent.handle(id, question, contexto=conversation_history[:])
                    print(f"📥 Lista recebida do agente: {lista_json}")

                    # 🔥 Enviando para o Firestore
                    try:
                        doc_ref = db.collection("dados_pacientes").document(id)
                        doc_ref.set({
                            "id": id,
                            "dados": lista_json
                        })
                        print("✅ Dados enviados ao Firebase com sucesso.")
                    except Exception as firebase_error:
                        print(f"❌ Erro ao enviar para o Firebase: {firebase_error}")

                    payload = SVCInput(features=lista_json)
                    resultado = await predict_cancer(payload)
                    
                    # 🔥 Enviando para o Firestore
                    try:
                        doc_ref = db.collection("resultados_predicao").document()
                        doc_ref.set({
                            "resultado": resultado["resultado"],
                            "probabilidade": resultado["probabilidade"],
                            "shap_values": resultado["shap_values"],
                            "features": lista_json
                        })
                        print("✅ Resultado da predição salvo com sucesso no Firestore.")
                    except Exception as firebase_error:
                        print(f"❌ Erro ao salvar resultado no Firestore: {firebase_error}")

                    print("🔍 Resultado da predição:", resultado)

                    modo_diagnostico_ativo = True


            if modo_diagnostico_ativo:
                # 🔍 Ativa o agente de diagnóstico
                response = await diag_agent.handle(id, resultado, contexto=conversation_history[:])
                if response is not None:
                    conversation_history.append({"role": "assistant", "content": response})
            
    except WebSocketDisconnect:
        print("❌ Cliente desconectado")
    except Exception as e:
        print(f"🔥 Erro inesperado: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[ERROR]")

#######################################################################################
#                                   DASHBOARDS                                        #
#######################################################################################

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from collections import Counter

router = APIRouter()

CSV_FILE_PATH = "./dataset.csv"  # aponte para o seu CSV na raiz

@router.get("/dashboard")
def get_dashboard_stats():
    # Carrega e normaliza colunas
    df = pd.read_csv(CSV_FILE_PATH)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.upper().str.replace(" ", "_")

    total = len(df) or 1

    # 1) Idade por faixa (0-9, 10-19, ...)
    age_bins = (df["AGE"] // 10) * 10
    age_counts = age_bins.value_counts().sort_index().to_dict()
    age_distribution = {f"{k}-{k+9}": v for k, v in age_counts.items()}
    age_percent      = {k: round(v/total*100, 1) for k, v in age_distribution.items()}

    # 2) Predição de câncer
    pred_series   = df["LUNG_CANCER"].map({"YES": "com câncer", "NO": "sem câncer"})
    pred_counts   = pred_series.value_counts().to_dict()
    pred_percent  = {k: round(v/total*100, 1) for k, v in pred_counts.items()}

    # 3) Gênero
    gender_series  = df["GENDER"].map({"M": "masculino", "F": "feminino"})
    gender_counts  = gender_series.value_counts().to_dict()
    gender_percent = {k: round(v/total*100, 1) for k, v in gender_counts.items()}

    return JSONResponse({
        "age_distribution":     age_distribution,
        "age_percent":          age_percent,
        "prediction_counts":    pred_counts,
        "prediction_percent":   pred_percent,
        "gender_counts":        gender_counts,
        "gender_percent":       gender_percent
    })

# No seu main.py, logo após importar o router:
app.include_router(router, prefix="/api")


# Rodar app manualmente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7000, reload=False)
