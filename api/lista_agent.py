import ast
import re
import json
import requests
import asyncio

class ListaAgent:
  def __init__(self, websocket, openai, isAsync):
      print('Inicializando ListaAgent')
      self.websocket = websocket
      self.isAsync = isAsync
      self.id = 0
      self.openai = openai

  def extrair_lista(self, texto):
      """Extrai lista do modelo a partir de um bloco markdown tipo ```lista ... ```"""
      padrao = r'```lista\s*(\[.*?\])\s*```'
      correspondencia = re.search(padrao, texto, re.DOTALL | re.IGNORECASE)

      if correspondencia:
          try:
              return ast.literal_eval(correspondencia.group(1))
          except Exception as e:
              print("Erro ao avaliar lista:", e)
              return None
      return None

  async def handle(self, id, question, contexto, data=None):
    try:
      self.id = id
      full_conversation = "\n\n".join([
         f"""{msg["role"]}: {msg["content"]}""" for msg in contexto
         ])
      print(full_conversation)
      
      prompt = f"""
        # 📦 Geração de LISTA a partir da Conversa

        ## 🎯 Objetivo

        Transformar as informações extraídas de uma conversa com o usuário em uma LISTA, respeitando regras específicas de conversão.

        ---

        ## 📌 Regras de Conversão

        - **Gênero**:
          - Masculino → `1`
          - Feminino → `0`

        - **Para os campos booleanos (sim/não)**, utilize as seguintes regras:
          - Respostas positivas (ex: "sim", "às vezes", "já", "costumo", "sim, ocasionalmente") → `1`
          - Respostas negativas (ex: "não", "nunca", "jamais", "de forma alguma") → `0`

        - **Campo "confirmado"**:
          - Se o usuário confirmou explicitamente que as informações estão corretas → `true`
          - Caso contrário → `false`

        ---
        obs:
          -os items que estarão na lista são GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY,PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING,ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY, CHEST_PAIN

        ## 🧾 Formato da lista Esperado

        ```lista
        [1, 22, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]

        

        Histórico da Conversa: 
        {full_conversation}


        Atenção retorne apenas a lista nenhuma informação a mais.
      """

      contexto.append({"role": "user", "content": prompt})

      response = self.openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=contexto
      )

      print(response.choices[0].message.get("content", ""))
      return self.extrair_lista(response.choices[0].message.get("content", ""))

    except Exception as e:
      print(e)

