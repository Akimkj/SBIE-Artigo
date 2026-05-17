import torch
from bert_score import BERTScorer
import pandas as pd
import os
from src import dataFormat
from typing import cast

def robertaLargeEvaluation(rawCandidate: list, rawGolden: list, outputPath: str | None):

    #Padroniza a estrutura de ambos os datasets com pydantic
    goldenset = dataFormat.QADataSet(data=rawGolden)
    candidateset = dataFormat.QADataSet(data=rawCandidate)

    #Carrega apenas uma vez as informações do modelo pré-treinado
    scorer = BERTScorer(
        model_type='roberta-large',
        lang='en',
        rescale_with_baseline=False,
        device=None,
    )


    results = []
    total = len(goldenset.data)
    print(f"\nIniciando avaliação de {total} questões usando BERTscore...")

    #Loop principal que percorre a lista goldenset
    for idx, gold in enumerate(goldenset.data, 1):
        print(f"Processando questão {idx}/{total}...", end='\r')

        #Procura o item do geminiset que tenha o mesmo id do item do goldenset
        candidateItem = next((item for item in candidateset.data if item.id == gold.id), None)

        if candidateItem:
            reference = gold.answer 
            candidate = candidateItem.answer
            
            #realiza o calculo com a referencia e o candidato
            P, R, F1 = scorer.score([candidate],[reference])

            #adiciona as informações do id e pergunta junto com seus resultados
            results.append({
                "id": candidateItem.id,
                "question": candidateItem.question,
                "bert_precision": round(cast(torch.Tensor, P).item(), 4),
                "bert_recall": round(cast(torch.Tensor, R).item(), 4),
                "bert_F1": round(cast(torch.Tensor, F1).item(), 4)
            })
        else:
            print(f"\nNão foi achado o ID correspondente para a questão {gold.id}...")
            
    print("\n\nAvaliação concluída com sucesso!")

    df = pd.DataFrame(results)
    if outputPath is None:
        outputPath = os.path.join("results", "csv", "avBert_who.csv")
    
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    df.to_csv(outputPath, index=False, encoding='utf-8')


    return df




        











