import re
import json
import requests
import asyncio

class ValidadorAgent:
  def __init__(self, websocket, openai, isAsync):
    print('Inicializando AnswerAgent')
    self.websocket = websocket
    self.isAsync = isAsync
    self.id = 0
    self.openai = openai

  async def handle(self, id, question, contexto, data=None):
    try:

      self.id = id
      full_conversation = "\n\n".join([f"""{msg["role"]}: {msg["content"]}""" for msg in contexto])
      # print(full_conversation)
      prompt = f"""
        # 🧠 Agente de Verificação de Informações

        ## 🎯 Objetivo do Agente

        Você é um agente responsável por verificar se **todas as informações abaixo foram respondidas** e se o **usuário confirmou o resumo das informações fornecidas**.

        ---

        ## ✅ Lista de Informações Obrigatórias

        Confirme se todas as seguintes questões foram respondidas na conversa:

        - **nome completo**
        - **idade**
        - **gênero**
        - **fumante**
        - **dedos amarelados**
        - **ansiedade**
        - **pressão de familiares ou amigos**
        - **doença crônica**
        - **fadiga com frequência**
        - **alergias**
        - **chiado ao respirar**
        - **consumo de álcool**
        - **tosse frequente**
        - **falta de ar**
        - **dificuldade para engolir**
        - **dor no peito**

        ---

        ## 📝 Confirmação Final

        Verifique também se o **usuário confirmou explicitamente** que as informações fornecidas estão corretas.

        Exemplos válidos de confirmação incluem:
        - "Sim, estão corretas"
        - "Confirmo"
        - "Pode seguir, está tudo certo"

        ---

        ## 🔎 Instruções de Resposta

        Com base na conversa completa, siga as regras abaixo:

        - Se **todas** as perguntas acima foram respondidas **e** o usuário **confirmou** o resumo das informações, responda **apenas** com: sim
        - Se **qualquer** pergunta estiver faltando **ou** o usuário **não tiver confirmado** o resumo, responda **apenas** com: não

        Histórico da Conversa: 
        {full_conversation}
      """

      contexto.append({"role": "user", "content": prompt})

      response = self.openai.ChatCompletion.create(
          model="gpt-4o-mini",
          messages=contexto
      )

      response = response.choices[0].message.get("content", "")

      if response is not None:
        result_text = response.strip().lower()
        print(result_text)
        result_bool = True if result_text == "sim" else False
        return result_bool

      return False

    except Exception as e:
      print(e)

