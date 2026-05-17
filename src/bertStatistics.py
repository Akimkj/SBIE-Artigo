import os
import numpy as np
import pandas as pd


# colunas Bert a serem analisadas
METRICS = ["bert_precision", "bert_recall", "bert_F1"]


def _calculate_statistics(model_name: str, csv_path: str) -> dict | None:
    if not os.path.exists(csv_path):
        print(f"  [WARNING] CSV não encontrado para {model_name}: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    result = {"model": model_name}

    for metric in METRICS:
        if metric not in df.columns:
            print(f"  [WARNING] Coluna '{metric}' não encontrada em {csv_path}")
            continue

        values = df[metric].dropna()
        result[f"{metric}_media"]   = round(float(np.mean(values)),   6)
        result[f"{metric}_mediana"] = round(float(np.median(values)), 6)
        result[f"{metric}_desvio"]    = round(float(np.std(values)),    6)

    return result


def generate_statistics(models: dict[str, str], output_path: str) -> None:

    print("\nCauculando BERT statistics...")

    rows = []
    for name, path in models.items():
        print(f"  Processando {name}...")
        stats = _calculate_statistics(name, path)
        if stats:
            rows.append(stats)

    if not rows:
        print("\n[ERROR] CSV não encontrado")
        return

    # Garante que a pasta destino exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n  Estatisticas salvas em: {output_path}!")