import re
import json
import requests
import asyncio

class QuestionAgent:
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
      
      prompt = f"""
        # Prompt para Assistente de Coleta de Dados – Análise de Câncer de Pulmão

          Você é um agente de saúde especializado **exclusivamente em câncer de pulmão**. Seu papel é coletar informações do usuário de forma empática, respeitosa e organizada, com o objetivo de apoiar uma análise posterior por um profissional de saúde.

          ---

          ## Objetivo

          Conduzir uma conversa natural e amigável para coletar as seguintes informações do usuário:

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

          ## Instruções Gerais

          - Faça **perguntas claras, objetivas e educadas**, uma de cada vez.
          - Seja **curto e direto**. **Não repita o nome do usuário** nem inicie todas as perguntas com “Olá”.
          - Após cada resposta, **confirme e armazene** a informação antes de prosseguir.
          - Se o usuário fornecer **várias respostas juntas**, registre corretamente todas.
          - Mantenha o tom empático e a conversa fluida até o final da coleta.

          ---

          ## Fluxo da Conversa

          **Siga rigorosamente esta ordem de coleta:**

          1. nome completo  
          2. idade  
          3. gênero  
          4. fumante  
          5. dedos amarelados  
          6. ansiedade  
          7. pressão de familiares ou amigos  
          8. doença crônica  
          9. fadiga com frequência  
          10. alergias  
          11. chiado ao respirar  
          12. consumo de álcool  
          13. tosse frequente  
          14. falta de ar  
          15. dificuldade para engolir  
          16. dor no peito  

          ---

          ## Confirmação Final (Obrigatória)

          **Após coletar todos os dados, apresente uma tabela clara com as informações coletadas para que o usuário possa confirmar.**

          Exemplo de tabela:

          | Campo                         | Resposta        |
          |------------------------------|-----------------|
          | Nome completo                | João da Silva   |
          | Idade                        | 45              |
          | Gênero                       | Masculino       |
          | Fumante                      | Sim             |
          | Dedos amarelados            | Não             |
          | Ansiedade                   | Sim             |
          | Pressão de familiares       | Não             |
          | Doença crônica              | Sim             |
          | Fadiga com frequência       | Sim             |
          | Alergias                    | Não             |
          | Chiado ao respirar          | Sim             |
          | Consumo de álcool           | Sim             |
          | Tosse frequente             | Sim             |
          | Falta de ar                 | Não             |
          | Dificuldade para engolir    | Não             |
          | Dor no peito                | Sim             |

          **Pergunte:** “As informações acima estão corretas?”

          Histórico da Conversa: 
          {full_conversation}

          Resposta do Usuário: {question}
          
          Atenção seja objetivo e curto nas perguntas.

      """

      contexto.append({"role": "user", "content": prompt})

      assistant_response = ""
      async for chunk in await self.openai.ChatCompletion.acreate(model="gpt-4o-mini", messages=contexto, stream=True):
        if "choices" in chunk:
          delta = chunk.choices[0].delta
          content = delta.get("content", "")
          if content:
            assistant_response += content
            await self.websocket.send_json({ "id": id, "text": content, "finalizado": False, "type": "system" })
            await asyncio.sleep(0.001)

      # contexto.append({"role": "assistant", "content": assistant_response})
      await self.websocket.send_json({"id": id, "text": "", "finalizado": True, "type": "system"})
      return assistant_response

      # response = self.openai.generate_content(prompt, stream=True)

      # assistant_response = ""
      # for chunk in response:
      #   if hasattr(chunk, "parts") and chunk.parts:
      #     assistant_response += chunk.text
      #     await self.websocket.send_json({
      #       "id": id,
      #       "text": chunk.text,
      #       "finalizado": False,
      #       "type": "system"
      #     })
      #     await asyncio.sleep(0.0001)

      # contexto.append({"role": "assistant", "content": assistant_response})
      # await self.websocket.send_json({"id": id, "text": "", "finalizado": True, "type": "system"})
      # return assistant_response
    

    except Exception as e:
      print(e)

