from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json

from modelo_predictor import SVCService
from predict import SVCInput
import openai

from question_agent import QuestionAgent
from validador_agent import ValidadorAgent
from lista_agent import ListaAgent
from diag_agent import DiagAgent
from starlette.websockets import WebSocketState

# API Key OpenAI
openai.api_key = ""

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

                  payload = SVCInput(features=lista_json)
                  resultado = await predict_cancer(payload)

                  print("🔍 Resultado da predição:", resultado)

                  # await websocket.send_text(json.dumps(resultado))
                  modo_diagnostico_ativo = True


            if modo_diagnostico_ativo:
              # 🔍 Ativa o agente de diagnóstic
              response = await diag_agent.handle(id, resultado, contexto=conversation_history[:])
              if response is not None:
                  conversation_history.append({"role": "assistant", "content": response})
            
    except WebSocketDisconnect:
        print("❌ Cliente desconectado")
    except Exception as e:
        print(f"🔥 Erro inesperado: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[ERROR]")

# Rodar app manualmente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7000, reload=False)
