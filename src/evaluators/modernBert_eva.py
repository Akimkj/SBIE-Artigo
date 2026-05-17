import sys
import torch

# Python 3.14+ não suporta torch.compile usado internamente pelo ModernBERT
if sys.version_info >= (3, 14):
    torch.compile = lambda fn=None, *args, **kwargs: (fn if fn is not None else lambda f: f)

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from src import dataFormat


MODEL_ID = "answerdotai/ModernBERT-large"


def _get_embeddings(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(device)
    inputs.pop("token_type_ids", None)  # ModernBERT não usa token_type_ids

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state.squeeze(0)

    # Remove tokens especiais (BOS/EOS/PAD) dos embeddings
    special_ids = set(tokenizer.all_special_ids)
    keep = [i for i, tid in enumerate(inputs["input_ids"].squeeze(0).tolist()) if tid not in special_ids]

    return hidden[keep] if keep else hidden


def _bertscore_prf(candidate, reference, tokenizer, model, device):
    # Extrai e normaliza os embeddings de cada texto
    emb_cand = F.normalize(_get_embeddings(candidate, tokenizer, model, device), p=2, dim=-1)
    emb_ref  = F.normalize(_get_embeddings(reference,  tokenizer, model, device), p=2, dim=-1)

    # Matriz de similaridade de cosseno e greedy matching
    sim_matrix = torch.mm(emb_cand, emb_ref.T)
    precision  = sim_matrix.max(dim=1).values.mean().item()
    recall     = sim_matrix.max(dim=0).values.mean().item()
    f1         = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def modernBertEvaluation(rawCandidate: list, rawGolden: list, outputPath: str | None):

    goldenset    = dataFormat.QADataSet(data=rawGolden)
    candidateset = dataFormat.QADataSet(data=rawCandidate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    print(f"Carregando modelo '{MODEL_ID}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.to(device)
    model.eval()
    print("Modelo carregado com sucesso!\n")

    results = []
    total   = len(goldenset.data)
    print(f"\nIniciando avaliação de {total} questões usando BERTScore (ModernBERT-large)...")

    for idx, gold in enumerate(goldenset.data, 1):
        print(f"Processando questão {idx}/{total}...", end="\r")

        candidateItem = next((item for item in candidateset.data if item.id == gold.id), None)

        if candidateItem:
            P, R, F1 = _bertscore_prf(candidateItem.answer, gold.answer, tokenizer, model, device)

            results.append({
                "id":             candidateItem.id,
                "question":       candidateItem.question,
                "bert_precision": round(P,  4),
                "bert_recall":    round(R,  4),
                "bert_F1":        round(F1, 4),
            })
        else:
            print(f"\nNão foi achado o ID correspondente para a questão {gold.id}...")

    print("\n\nAvaliação concluída com sucesso!")

    df = pd.DataFrame(results)

    if outputPath is None:
        outputPath = os.path.join("results", "csv", "avBert_who.csv")

    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    df.to_csv(outputPath, index=False, encoding="utf-8")

    return df