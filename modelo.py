from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import random

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

# Aqui você define quem pode acessar sua API
origins = [
    "http://localhost:3000",  # Frontend local
    "http://127.0.0.1:3000",  # Outra forma de chamar localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Você também pode usar ["*"] para liberar tudo (não recomendado em produção)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("SVC.pkl", "rb") as f:
    modelo = pickle.load(f)

mensagens_sem_cancer = [
    "Boa notícia! Os sinais indicam ausência de câncer de pulmão. 😊",
    "Tudo certo por aqui! Nenhum indício de câncer foi detectado. ✨",
    "Ufa! De acordo com a análise, está tudo limpo. 💪",
    "Parabéns! Os resultados sugerem que você está saudável. 🫁💚",
    "Tranquilo! Nada de anormal foi identificado no momento. 🧘‍♂️",
    "Ótimas notícias: sem sinais de câncer de pulmão por aqui! 🙌",
    "A análise foi concluída e o resultado é positivo: você está bem! 🥳",
    "Nada preocupante foi encontrado. Continue se cuidando! 🍀",
    "Seu pulmão parece saudável de acordo com os dados. 💨💙",
    "Os sinais não indicam câncer. Respire aliviado! 😌",
    "O modelo não encontrou indícios preocupantes. Fique tranquilo! 🌟",
    "Que alívio! A previsão não mostra nenhum risco detectável. 🎉",
    "Saúde em dia! Nenhuma anormalidade foi encontrada. 🏃‍♂️🍎",
    "Fique em paz, os dados apontam que está tudo bem. 🙏",
    "Resultado excelente: sem evidências de câncer. 💚🫁",
    "Nada a temer no momento. Tudo parece estar nos conformes. 🧑‍⚕️",
    "O diagnóstico não indica câncer. Continue atento à sua saúde! 🕊️",
    "Boas novas! Não encontramos sinais de câncer pulmonar. 🌿",
    "Seu pulmão está aparentemente saudável. Mantenha bons hábitos! 🌞",
    "Nada detectado! Continue cuidando bem da sua saúde. 💖"
]

mensagens_com_cancer = [
    "Atenção: os sinais indicam possíveis indícios de câncer de pulmão. 🩺",
    "Recomendamos buscar orientação médica. O modelo detectou sinais de alerta. ⚠️",
    "Cuidado! A análise sugere um possível caso de câncer. Consulte um especialista. 🙏",
    "É importante investigar: os dados apontam risco para câncer de pulmão. 🧬",
    "Alerta: há sinais consistentes com um quadro de câncer. Procure um profissional. 🏥",
    "O modelo identificou indícios preocupantes. Não deixe de buscar apoio médico. 🤝",
    "A análise apontou sinais de risco. Marque uma consulta o quanto antes. 🩻",
    "Prevenção é essencial: o resultado sugere a necessidade de atenção médica. 💡",
    "Sinais compatíveis com câncer foram detectados. Cuide-se e procure ajuda. 🧑‍⚕️",
    "Atenção redobrada: há fortes indícios que exigem investigação médica. 🛑",
    "Recomendamos procurar um profissional para uma avaliação mais detalhada. 💬",
    "Sinais críticos encontrados. Não adie: procure um especialista. 📞",
    "Nosso sistema detectou um possível problema. Consulte seu médico. 🧠",
    "O resultado sugere que algo não está certo. Um exame clínico é essencial. 📋",
    "Importante: os sinais encontrados merecem atenção imediata. 🚨",
    "Risco identificado. Marque uma consulta e cuide de você! 💗",
    "Há sinais que podem indicar um problema. É hora de buscar orientação. 🗓️",
    "O modelo prevê possíveis sinais de câncer. Não hesite em investigar. 🩺",
    "Resultado alerta para riscos. Priorize sua saúde e busque ajuda. 🙌",
    "Sinais detectados que podem representar perigo. Aja com responsabilidade. 💭"
]

class EntradaDados(BaseModel):
    GENDER: int
    AGE: int
    SMOKING: int
    YELLOW_FINGERS: int
    ANXIETY: int
    PEER_PRESSURE: int
    CHRONIC_DISEASE: int
    FATIGUE: int
    ALLERGY: int
    WHEEZING: int
    ALCOHOL_CONSUMING: int
    COUGHING: int
    SHORTNESS_OF_BREATH: int
    SWALLOWING_DIFFICULTY: int
    CHEST_PAIN: int

@app.post("/predict")
def prever_diagnostico(dados: EntradaDados):
    entrada = np.array([list(dados.dict().values())])
    print(entrada)
    pred = modelo.predict(entrada)[0]

    if pred == 1:
        resultado = random.choice(mensagens_com_cancer)
    else:
        resultado = random.choice(mensagens_sem_cancer)
    return {"previsao": int(pred), "resultado": resultado}
