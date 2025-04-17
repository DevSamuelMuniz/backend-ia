import random
from fastapi import APIRouter, UploadFile, File
from api.models.chat import ChatRequest, ChatResponse
from api.models.predict import SVCInput
from api.services.gemini_service import GeminiChat
from api.services.svc_service import SVCService

router = APIRouter()
gemini = GeminiChat()
svc = SVCService()


baixo_risco_respostas = [
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

alto_risco_respostas = [
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

@router.post("/", response_model=ChatResponse)
async def chat_with_bot(payload: ChatRequest):
    return gemini.chat(payload)

@router.post("/predict/")
async def predict_cancer(payload: SVCInput):
    result = svc.predict(payload)
    
    if result["prediction"] == 1:
        resposta = random.choice(alto_risco_respostas)
    else:
        resposta = random.choice(baixo_risco_respostas)
    
    return {
        "resultado": resposta,
    }

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = gemini.process_file(file.filename, content)
    return {"context_text": text}
