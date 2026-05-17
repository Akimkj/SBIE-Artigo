import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import rotations

# Pasta onde as imagens geradas serão salvas
OUTPUT_DIR = os.path.join("results", "graphs")

# Cores fixas por modelo para manter consistência entre gráficos
MODEL_COLORS = {
    "Gemini":   "#4285F4",
    "Claude":   "#CC785C",
    "OpenAI":   "#10A37F",
    "DeepSeek": "#7B5EA7",
}

# Nomes amigáveis para exibição nos gráficos
METRIC_NAMES = {
    "precision": "Precision",
    "recall":    "Recall",
    "F1":        "F1-Score",
}

STAT_NAMES = {
    "media":   "Média",
    "mediana": "Mediana",
    "desvio":  "Desvio",
}

# Coluna no CSV individual de cada modelo para o question heatmap
HEATMAP_METRIC_COL = {
    "precision": "bert_precision",
    "recall":    "bert_recall",
    "F1":        "bert_F1",
}

# Pasta padrão dos CSVs por questão
CSV_DIR = os.path.join("results", "csv")


def _get_color(model, index):
    """Retorna a cor do modelo ou uma cor padrão pelo índice."""
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    return MODEL_COLORS.get(model, fallback[index % len(fallback)])


def _save(fig, filename):
    """Cria a pasta de saída (se necessário) e salva o gráfico."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


# GRÁFICO 1 — Barras agrupadas

def plot_bar(df, metrics, stats):
    models = df["model"].tolist()
    n_models = len(models)
    x = np.arange(len(metrics))
    width = 0.7 / n_models

    fig, axes = plt.subplots(1, len(stats), figsize=(10 * len(stats), 6), squeeze=False)

    for col, stat in enumerate(stats):
        ax = axes[0][col]
        for i, model in enumerate(models):
            offset = (i - n_models / 2 + 0.5) * width
            values = [df.loc[df["model"] == model, f"bert_{m}_{stat}"].values[0] for m in metrics]

            bars = ax.bar(x + offset, values, width * 0.9,
                          label=model, color=_get_color(model, i))

            # Anota o valor em cima de cada barra
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_NAMES[m] for m in metrics], fontsize=12)
        ax.set_ylabel(f"Score ({STAT_NAMES[stat]})")
        ax.set_title(f"Bar Chart — {STAT_NAMES[stat]}")
        ax.legend(title="Model")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("BERT Metrics — Grouped Bar Chart", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "bar_chart.png")


# GRÁFICO 2 — Box plot
def plot_box(df, metrics, stats):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6), sharey=False)

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, (_, row) in enumerate(df.iterrows()):
            model = row["model"]
            mean  = row[f"bert_{metric}_media"]
            std   = row[f"bert_{metric}_desvio"]
            med   = row[f"bert_{metric}_mediana"]
            color = _get_color(model, i)

            # Caixa mean ± std (sempre desenhada como base do box plot)
            ax.bar(i, 2 * std, bottom=mean - std, width=0.5,
                   color=color, alpha=0.5, edgecolor=color, linewidth=1.5)

            # Traço da mediana (exibido se "mediana" foi selecionada)
            if "mediana" in stats:
                ax.plot([i - 0.25, i + 0.25], [med, med], color=color, linewidth=2.5)

            # Ponto da média (exibido se "media" foi selecionada)
            if "media" in stats:
                ax.scatter([i], [mean], color="white", edgecolors=color, s=60, zorder=5)

            # Whiskers ±2σ (exibidos se "desvio" foi selecionado)
            if "desvio" in stats:
                ax.plot([i, i], [mean - 2 * std, mean - std], color=color, lw=1.2, ls="--")
                ax.plot([i, i], [mean + std, mean + 2 * std], color=color, lw=1.2, ls="--")

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["model"].tolist(), rotation=20, fontsize=10)
        ax.set_title(METRIC_NAMES[metric])
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legenda descrevendo o que está visível
    shown = []
    if "media"   in stats: shown.append("○ = Mean")
    if "mediana" in stats: shown.append("— = Median")
    if "desvio"  in stats: shown.append("box = ±SD  |  whiskers = ±2SD")
    fig.suptitle("BERT Metrics — Box Plot\n" + "   ".join(shown), fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "box_plot.png")


# GRÁFICO 3 — Radar
def plot_radar(df, metrics, stats):
    if len(metrics) < 3:
        print("  [Aviso] Radar requer pelo menos 3 métricas. Pulando...")
        return

    labels = [METRIC_NAMES[m] for m in metrics]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(stats), figsize=(7 * len(stats), 7),
                             subplot_kw={"polar": True})

    if len(stats) == 1:
        axes = [axes]

    for ax, stat in zip(axes, stats):
        for i, (_, row) in enumerate(df.iterrows()):
            model  = row["model"]
            values = [row[f"bert_{m}_{stat}"] for m in metrics]
            values += values[:1]
            color  = _get_color(model, i)

            ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
        ax.set_title(f"Radar — {STAT_NAMES[stat]}", fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    fig.suptitle("BERT Metrics — Radar Chart", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "radar_chart.png")


# GRÁFICO 4 — Heatmap
def plot_heatmap(df, metrics, stats):
    col_labels = [f"{METRIC_NAMES[m]} {STAT_NAMES[s]}" for m in metrics for s in stats]
    data = []
    for _, row in df.iterrows():
        data.append([row[f"bert_{m}_{s}"] for m in metrics for s in stats])

    matrix = np.array(data)
    models = df["model"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.3), max(4, len(models) * 1.3)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    # Anota o valor em cada célula
    for r in range(len(models)):
        for c in range(len(col_labels)):
            ax.text(c, r, f"{matrix[r, c]:.4f}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if matrix[r, c] > 0.85 else "black")

    ax.set_title("BERT Metrics — Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "heatmap.png")


# GRÁFICO 5 — Line chart
def plot_line(df, metrics, stats):
    if len(metrics) < 2:
        print("  [Aviso] Line chart requer pelo menos 2 métricas. Pulando...")
        return

    # Estilos de linha para diferenciar as estatísticas visualmente
    stat_styles = {"media": "-", "mediana": "--", "desvio": ":"}

    labels = [METRIC_NAMES[m] for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (_, row) in enumerate(df.iterrows()):
        model = row["model"]
        color = _get_color(model, i)

        for stat in stats:
            values = [row[f"bert_{m}_{stat}"] for m in metrics]
            ls     = stat_styles[stat]
            # Rótulo só na primeira estatística para não duplicar na legenda
            label = model if stat == stats[0] else None

            ax.plot(range(len(metrics)), values, "o" + ls,
                    label=label, color=color, linewidth=2, markersize=7, alpha=0.85)

            # Anota o valor e o nome da estatística em cada ponto
            for xi, v in enumerate(values):
                ax.annotate(f"{v:.4f}\n({STAT_NAMES[stat]})",
                            (xi, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color=color, alpha=0.8)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("BERT Metrics — Line Chart")
    ax.legend(title="Model")
    ax.grid(linestyle="--", alpha=0.4)

    # Explica os estilos de linha no eixo X
    style_legend = "  ".join([f"{stat_styles[s]} = {STAT_NAMES[s]}" for s in stats])
    ax.set_xlabel(f"Line style:  {style_legend}", fontsize=9)

    fig.tight_layout()
    _save(fig, "line_chart.png")


# GRÁFICO 6 — Question Heatmap (modelo × questão)
def plot_question_heatmap(csv_dir, metric):
    """
    Lê os CSVs individuais de cada LLM em csv_dir e gera um heatmap
    onde linhas = modelos e colunas = questões, colorido pela métrica escolhida.
    """
    col          = HEATMAP_METRIC_COL.get(metric, "bert_F1")
    metric_label = METRIC_NAMES.get(metric, metric)

    model_data = {}
    questions_ids = None
    for fname in sorted(os.listdir(csv_dir)):
        if fname.startswith("avBert_") and fname.endswith(".csv"):
            raw_name   = fname.replace("avBert_", "").replace(".csv", "")
            model_name = raw_name.capitalize()
            fpath      = os.path.join(csv_dir, fname)
            df_m       = pd.read_csv(fpath)
            if col in df_m.columns:
                model_data[model_name] = df_m[col].values

                if questions_ids is None:
                    id_col = next(
                        (c for c in df_m.columns if c.lower() in ("id")), None
                    )
                    question_ids = df_m[id_col].tolist() if id_col else list(range(len(df_m)))

    if not model_data:
        print(f"  [Aviso] Nenhum CSV avBert_*.csv encontrado em {csv_dir}. Pulando...")
        return

    models = list(model_data.keys())
    n_q    = len(next(iter(model_data.values())))


    matrix     = np.array([model_data[m] for m in models])


    fig_w = max(16, n_q * 0.20)
    fig_h = max(3,  len(models) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(metric_label, fontsize=11)

    # Eixo Y — modelos
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_ylabel("Model", fontsize=12)

    # Eixo X — questões (tick a cada ~5 questões para não poluir)
    xtick_pos = list(range(n_q))
    xtick_lbl = [str(question_ids[i]) for i in xtick_pos]

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=8, rotation=90)
    ax.set_xlabel("Question", fontsize=12)

    # Grade fina entre células
    ax.set_xticks(np.arange(-0.5, n_q, 1),        minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Separadores horizontais entre modelos
    for y in np.arange(0.5, len(models) - 0.5):
        ax.axhline(y, color="white", linewidth=1.5)

    ax.set_title(f"{metric_label} Heatmap by Question and Model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"question_heatmap_{metric + "v1"}.png")


# ──────────────────────────────────────────────────────────
# DISPATCHER — chamado pelo menu da main
# ──────────────────────────────────────────────────────────

CHART_OPTIONS = {
    1: ("Bar Chart",        plot_bar),
    2: ("Box Plot",         plot_box),
    3: ("Radar Chart",      plot_radar),
    4: ("Heatmap",          plot_heatmap),
    5: ("Line Chart",       plot_line),
    6: ("Question Heatmap", plot_question_heatmap),
}

METRIC_OPTIONS = {
    1: ["precision", "recall", "F1"],
    2: ["precision"],
    3: ["recall"],
    4: ["F1"],
    5: ["precision", "F1"],
    6: ["recall", "F1"],
}

STAT_OPTIONS = {
    1: ["media", "mediana", "desvio"],  # todas
    2: ["media"],
    3: ["mediana"],
    4: ["desvio"],
    5: ["media", "mediana"],
    6: ["media", "desvio"],
    7: ["mediana", "desvio"],
}

# Opções de métrica exclusivas para o Question Heatmap
HEATMAP_METRIC_OPTIONS = {
    1: "precision",
    2: "recall",
    3: "F1",
}


def run_charts(stats_csv_path, csv_dir=None):
    """
    Exibe o submenu de gráficos e gera o gráfico escolhido.

    Parâmetros
    ----------
    stats_csv_path : str
        Caminho do CSV de estatísticas gerado por generate_statistics().
    csv_dir : str, optional
        Pasta com os CSVs individuais de cada LLM (avBert_*.csv).
        Se None, usa CSV_DIR (results/csv).
    """
    if csv_dir is None:
        csv_dir = CSV_DIR

    # ── Seleção de gráfico ──
    print("\n--- Tipo de Gráfico ---")
    for key, (name, _) in CHART_OPTIONS.items():
        print(f"{key} - {name}")
    print("7 - Todos os gráficos")
    c_op = int(input("Escolha o gráfico: "))

    # ── Question Heatmap: fluxo próprio (métrica apenas, sem estatísticas) ──
    if c_op == 6:
        print("\n--- Métrica para o Question Heatmap ---")
        for k, m in HEATMAP_METRIC_OPTIONS.items():
            print(f"{k} - {METRIC_NAMES[m]}")
        hm_op  = int(input("Escolha a métrica: "))
        metric = HEATMAP_METRIC_OPTIONS.get(hm_op, "F1")
        print(f"\nGerando Question Heatmap ({METRIC_NAMES[metric]})...")
        plot_question_heatmap(csv_dir, metric)
        return

    # ── Para os demais gráficos: precisa do CSV de estatísticas ──
    if not os.path.exists(stats_csv_path):
        print(f"\n[Erro] Arquivo de estatísticas não encontrado: {stats_csv_path}")
        print("Execute a opção 4 primeiro para gerar as estatísticas.\n")
        return

    df = pd.read_csv(stats_csv_path)

    # ── Seleção de métricas ──
    print("\n--- Métricas ---")
    print("1 - Todas (Precision, Recall, F1)")
    print("2 - Apenas Precision")
    print("3 - Apenas Recall")
    print("4 - Apenas F1")
    print("5 - Precision + F1")
    print("6 - Recall + F1")
    m_op    = int(input("Escolha as métricas: "))
    metrics = METRIC_OPTIONS.get(m_op, METRIC_OPTIONS[1])

    # ── Seleção de estatísticas ──
    print("\n--- Estatísticas ---")
    print("1 - Todas (Média, Mediana, Desvio Padrão)")
    print("2 - Apenas Média")
    print("3 - Apenas Mediana")
    print("4 - Apenas Desvio Padrão")
    print("5 - Média + Mediana")
    print("6 - Média + Desvio Padrão")
    print("7 - Mediana + Desvio Padrão")
    s_op  = int(input("Escolha as estatísticas: "))
    stats = STAT_OPTIONS.get(s_op, STAT_OPTIONS[1])

    print()
    if c_op == 7:
        # Gera todos os gráficos de estatísticas
        stats_charts = {k: v for k, v in CHART_OPTIONS.items() if k != 6}
        for _, (name, func) in stats_charts.items():
            print(f"Gerando {name}...")
            func(df, metrics, stats)
        # Question Heatmap com F1 por padrão ao gerar tudo
        print("Gerando Question Heatmap (F1)...")
        plot_question_heatmap(csv_dir, "F1")
    elif c_op in CHART_OPTIONS:
        name, func = CHART_OPTIONS[c_op]
        print(f"Gerando {name}...")
        func(df, metrics, stats)
    else:
        print("Opção inválida.")