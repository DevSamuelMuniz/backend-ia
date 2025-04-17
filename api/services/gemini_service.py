import os, json
from dotenv import load_dotenv
import google.generativeai as genai
from api.models.chat import ChatRequest, ChatResponse
from api.utils.pdf_reader import extract_text_from_pdf

load_dotenv()
genai.configure(api_key=os.getenv("APIKEY_GEMINI"))

class GeminiChat:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def _load_history(self, user_id):
        try:
            with open("api/db/history.json", "r") as f:
                history = json.load(f)
            return history.get(user_id, [])
        except:
            return []

    def _save_history(self, user_id, history):
        try:
            with open("api/db/history.json", "r") as f:
                all_histories = json.load(f)
        except:
            all_histories = {}

        all_histories[user_id] = history
        with open("api/db/history.json", "w") as f:
            json.dump(all_histories, f)

    def chat(self, payload: ChatRequest) -> ChatResponse:
        history = self._load_history(payload.user_id)
        chat = self.model.start_chat(history=history)

        response = chat.send_message(payload.message)
        history.append({"role": "user", "parts": [payload.message]})
        history.append({"role": "model", "parts": [response.text]})

        self._save_history(payload.user_id, history)
        return ChatResponse(response=response.text)

    def process_file(self, filename, content: bytes) -> str:
        if filename.endswith(".pdf"):
            return extract_text_from_pdf(content)
        elif filename.endswith(".txt"):
            return content.decode("utf-8")
        else:
            return "Formato de arquivo não suportado."
