import json, time
from operator import attrgetter
from . import dataFormat, utils
from src.count_tokens import count_tokens
from src.services.gemini_service import callApiGemini
from src.services.claude_service import callApiClaude
from src.services.deepseek_service import callApiDeepseek
from src.services.openai_service import callApiOpenai
from google.genai import errors
from pydantic import ValidationError


TIME_BETWEEN_CALLS = 10


def process_questions(goldenSet: list, pathCandidate: str, model_engine: str):

    #1. LÓGICA DE PERSISTÊNCIA
    #Carrega os dados existentes no caminho informado  
    rawData = utils.loadData(pathCandidate)

    if not rawData: #se não existir dados, carrega um dataset vazio
        print("Criando novo dataset vazio...")
        candidateSet = dataFormat.QADataSet()
    else:
        try: #Se existir dados, carrega um dataset com os dados ja existentes
            candidateSet = dataFormat.QADataSet(data=rawData)
        except ValidationError:
            print("Arquivo com formato inválido, resetando...")
            candidateSet = dataFormat.QADataSet()
    

    #parte da lógica de persistencia
    processedIDs = {item.id for item in candidateSet.data}

    #2. Loop principal do algoritmo
    for i, rawItem in enumerate(goldenSet):

        currentID = rawItem.get('id') #id do item atual
        questionText = rawItem.get('question') #questao do item atual
        tokens_answer_stackOverflow = count_tokens(rawItem.get('answer'), model_engine) # quantidade de tokens da resposta do item atual

        #Pula se o ID já foi processado anteriormente
        if (currentID in processedIDs):
            print(f"ID {currentID} já foi processado")
            continue


        print(f"{currentID} Processando com {model_engine}...")

        try:
            # 3. Chamada da Api
            

            if model_engine == "claude":
                QApair = callApiClaude(currentID, questionText, tokens_answer_stackOverflow)
            elif model_engine == "deepseek":
                QApair = callApiDeepseek(currentID, questionText, tokens_answer_stackOverflow)
            elif model_engine == "gemini":
                QApair = callApiGemini(currentID, questionText, tokens_answer_stackOverflow)
            else:
                QApair = callApiOpenai(currentID, questionText, tokens_answer_stackOverflow)
                

            clean_json = QApair.strip().replace("```json", "").replace("```", "")

            # 4. Validação e Parsing
            # carrega a string do json limpa
            QApair_dict = json.loads(clean_json)

            QApair_dict['id'] = currentID

            #validação e parsing por meio do pydantic
            validatedEntry = dataFormat.QAPairModel(**QApair_dict)

            candidateSet.data.append(validatedEntry) 
            processedIDs.add(currentID)
            print(f"ID {currentID} Feito com sucesso")


        except (json.decoder.JSONDecodeError, ValidationError, errors.APIError, Exception) as e:
            print(f"ID {currentID}: Erro ao processar: {e}")


        # O salvamento no dataset ocorre a cada 5 pares (sucesso ou falha)
        if (i + 1) % 5 == 0:
            print(f"Salvando dados no {pathCandidate}")
            with open(pathCandidate, 'w', encoding='utf-8') as f:
                f.write(candidateSet.model_dump_json(indent=2))
        
        time.sleep(TIME_BETWEEN_CALLS)

    #Ordenando todos os dados por meio do ID
    candidateSet.data.sort(key=attrgetter('id'))

    #Salvamento final do dataset
    with open(pathCandidate, 'w', encoding='utf-8') as f:
        f.write(candidateSet.model_dump_json(indent=2))