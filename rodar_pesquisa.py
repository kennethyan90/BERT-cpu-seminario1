import numpy as np
from research.benchmark import run_sgns, summarize, pareto, Result 

# Definição dos parâmetros do experimento (Questão 15)
dimensoes = [8, 32, 64]   
negativos = [2, 5, 10]    
seeds = [0, 1, 2]         

resultados = []

print("=== INICIANDO VARREDURA DA QUESTÃO 15 ===")

# Loop pelos hiperparâmetros
for d in dimensoes:
    for k in negativos:
        print(f"\nTestando: Dimensão D={d}, Negativos K={k}...")

        # Listas para guardar as métricas das 3 sementes
        scores_semente = []
        flops_semente = []
        seconds_semente = []

        for s in seeds:
            # Executa a função nativa do repositório
            res = run_sgns(dim=d, neg_k=k, epochs=100, seed=s)

            # Acessando como atributos do objeto Result (dataclass)
            scores_semente.append(res.score)
            flops_semente.append(res.train_flops)
            seconds_semente.append(res.seconds)

        # Calcular as médias e desvios padrão para as sementes
        mean_score = np.mean(scores_semente)
        std_score = np.std(scores_semente)
        mean_flops = np.mean(flops_semente)
        mean_seconds = np.mean(seconds_semente)

        resultados.append(Result(
            name=f"D{d}_K{k}",
            seed=0,  
            score=mean_score,
            train_flops=int(mean_flops),
            infer_flops=0,
            seconds=mean_seconds,
            extra={"n_params": (len(res.extra["word2id"]) * d * 2)}  
        ))

# Exibir a tabela final
print("\n=== TABELA FINAL DE RESULTADOS ===")
print(summarize(resultados, baseline="D32_K5"))

import matplotlib.pyplot as plt
pareto(resultados, cost="train_flops", title="Word2Vec: Silhouette vs Custo de Treino (FLOPs)")
plt.savefig("fronteira_pareto_q15.png", dpi=300)
print("\n[OK] Gráfico 'fronteira_pareto_q15.png' salvo com sucesso na raiz!")
