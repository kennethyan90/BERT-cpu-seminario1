import time
import numpy as np
import matplotlib.pyplot as plt
from research.benchmark import (
    run_sgns, summarize, pareto, Result,
    REPRESENTATIVE, representative_vectors, silhouette_samples
)

print("=== INICIANDO EXPERIMENTO COMPLETO - QUESTÃO 15 (BERT-CPU) ===")
inicio_global = time.time()

# -------------------------------------------------------------------------
# PARTE 1: Grid Principal (D x K) incluindo limites inferiores (D=2, 4)
# -------------------------------------------------------------------------
dimensoes = [2, 4, 8, 16, 32, 64]
negativos = [2, 5, 10]
seeds = [0, 1, 2]

resultados_grid = []
modelos_para_inspecao = {}  # Guarda seed=0 (K=2) para inspeção qualitativa

total_runs_grid = len(dimensoes) * len(negativos) * len(seeds)
run_atual = 0

for d in dimensoes:
    for k in negativos:
        cfg_name = f"D{d}_K{k}"
        print(f"\n>>> Configuração: {cfg_name}")
        for s in seeds:
            run_atual += 1
            res = run_sgns(dim=d, neg_k=k, epochs=100, seed=s, tie_tables=False, name=cfg_name)
            print(f"  [{run_atual}/{total_runs_grid}] seed={s} | score={res.score:+.4f} | "
                  f"GFLOPs={res.gflops:.2f} | params={res.extra['n_params']:,} | {res.seconds:.1f}s")
            
            resultados_grid.append(res)
            
            if s == 0 and k == 2 and d in (2, 4, 8, 32, 64):
                modelos_para_inspecao[cfg_name] = res

# -------------------------------------------------------------------------
# PARTE 2: Experimento de Weight Tying (tie_tables=True - Press & Wolf, 2017)
# -------------------------------------------------------------------------
print("\n=== PARTE 2: TESTANDO WEIGHT TYING (tie_tables=True) ===")
dims_tying = [8, 32, 64]
negs_tying = [2, 5]
resultados_tying = []

for d in dims_tying:
    for k in negs_tying:
        cfg_name = f"D{d}_K{k}_tied"
        print(f"\n>>> Configuração (Tied): {cfg_name}")
        for s in seeds:
            res = run_sgns(dim=d, neg_k=k, epochs=100, seed=s, tie_tables=True, name=cfg_name)
            print(f"  seed={s} | score={res.score:+.4f} | GFLOPs={res.gflops:.2f} | "
                  f"params={res.extra['n_params']:,} | {res.seconds:.1f}s")
            resultados_tying.append(res)

# -------------------------------------------------------------------------
# PARTE 3: Funções Auxiliares para Sanity Check (Per-word & Nearest Neighbours)
# -------------------------------------------------------------------------
def gerar_relatorio_qualitativo(res_obj, top_k=5):
    model = res_obj.extra["model"]
    word2id = res_obj.extra["word2id"]
    id2word = res_obj.extra["id2word"]
    W = model.in_emb.weight.data
    
    vecs, labels, words = representative_vectors(W, word2id)
    coeffs = silhouette_samples(vecs, labels)
    
    linhas = []
    linhas.append(f"--- Detalhamento por Palavra e Vizinhos: {res_obj.name} (seed=0, score={res_obj.score:+.4f}) ---")
    
    nomes_clusters = list(REPRESENTATIVE.keys())
    for ci, cname in enumerate(nomes_clusters):
        mask = (labels == ci)
        c_mean = coeffs[mask].mean()
        palavras_cluster = [f"{w} ({coeffs[i]:+.3f})" for i, w in enumerate(words) if labels[i] == ci]
        linhas.append(f"  Cluster '{cname}' (média = {c_mean:+.3f}): " + ", ".join(palavras_cluster))
    
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    linhas.append("  Vizinhos mais próximos (Cosine Similarity):")
    for target_w in ["square", "circle", "woman", "space", "line"]:
        if target_w not in word2id:
            continue
        wid = word2id[target_w]
        sims = Wn @ Wn[wid]
        order = [j for j in np.argsort(-sims) if j != wid][:top_k]
        vizinhos_str = ", ".join(f"{id2word[j]} ({sims[j]:+.2f})" for j in order)
        linhas.append(f"    {target_w:<10} -> {vizinhos_str}")
    return "\n".join(linhas)

# -------------------------------------------------------------------------
# PARTE 4: Gerar e Salvar Tabelas e Relatório Completo
# -------------------------------------------------------------------------
tabela_principal = summarize(resultados_grid, baseline="D32_K5")
combinados_tying = [r for r in resultados_grid if r.name in ("D8_K2", "D8_K5", "D32_K2", "D32_K5", "D64_K2", "D64_K5")] + resultados_tying
tabela_tying = summarize(combinados_tying, baseline="D32_K5")

relatorio_qualitativo = "\n\n".join(
    gerar_relatorio_qualitativo(modelos_para_inspecao[k])
    for k in ["D2_K2", "D4_K2", "D8_K2", "D32_K2", "D64_K2"]
    if k in modelos_para_inspecao
)

texto_final = (
    "=====================================================================\n"
    "1. TABELA PRINCIPAL: VARREDURA D x K (3 SEEDS POR CONFIGURAÇÃO)\n"
    "=====================================================================\n"
    f"{tabela_principal}\n\n"
    "=====================================================================\n"
    "2. EFEITO DO WEIGHT TYING (REDUÇÃO DE 50% DOS PARÂMETROS LIVRES)\n"
    "=====================================================================\n"
    f"{tabela_tying}\n\n"
    "=====================================================================\n"
    "3. SANITY CHECK: SILHUETA POR PALAVRA E VIZINHOS MAIS PRÓXIMOS\n"
    "=====================================================================\n"
    f"{relatorio_qualitativo}\n"
)

print("\n" + texto_final)

with open("Resultados_Completos_Q15.txt", "w", encoding="utf-8") as f:
    f.write(texto_final)

# -------------------------------------------------------------------------
# PARTE 5: Gráficos para os Slides e Relatório (300 DPI)
# -------------------------------------------------------------------------
ax = pareto(resultados_grid, cost="train_flops", title="Word2Vec (Flatland): Silhouette vs Custo de Treino (FLOPs)")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fronteira_pareto_q15.png", dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for k in negativos:
    means, stds = [], []
    for d in dimensoes:
        scores_dk = [r.score for r in resultados_grid if r.name == f"D{d}_K{k}"]
        means.append(np.mean(scores_dk))
        stds.append(np.std(scores_dk, ddof=1))
    ax.errorbar(dimensoes, means, yerr=stds, marker="o", capsize=4, lw=2, label=f"K = {k} negativos")

ax.set_xscale("log", base=2)
ax.set_xticks(dimensoes)
ax.set_xticklabels([str(d) for d in dimensoes])
ax.set_xlabel("Dimensão da Tabela de Embedding (D) [escala log2]")
ax.set_ylabel("Silhouette Score (Média ± DP)")
ax.set_title("Trade-off de Viés-Variância: Impacto da Dimensão D (Yin & Shen, 2018)")
ax.grid(True, ls="--", alpha=0.5)
ax.legend()
fig.tight_layout()
plt.savefig("curva_vies_variancia_D.png", dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
x_pos = np.arange(len(dims_tying))
width = 0.35

untied_means = [np.mean([r.score for r in resultados_grid if r.name == f"D{d}_K2"]) for d in dims_tying]
untied_stds  = [np.std([r.score for r in resultados_grid if r.name == f"D{d}_K2"], ddof=1) for d in dims_tying]
tied_means   = [np.mean([r.score for r in resultados_tying if r.name == f"D{d}_K2_tied"]) for d in dims_tying]
tied_stds    = [np.std([r.score for r in resultados_tying if r.name == f"D{d}_K2_tied"], ddof=1) for d in dims_tying]

ax.bar(x_pos - width/2, untied_means, width, yerr=untied_stds, capsize=4, label="Duas Tabelas (Padrão: 2·V·D params)")
ax.bar(x_pos + width/2, tied_means, width, yerr=tied_stds, capsize=4, label="Tabelas Amarradas (Tied: 1·V·D params)")

ax.set_xticks(x_pos)
ax.set_xticklabels([f"D = {d}" for d in dims_tying])
ax.set_ylabel("Silhouette Score (Média ± DP) [K=2]")
ax.set_title("Impacto do Weight Tying (Press & Wolf, 2017) na Qualidade Semântica")
ax.axhline(0, color="black", lw=0.8)
ax.grid(True, axis="y", ls="--", alpha=0.5)
ax.legend()
fig.tight_layout()
plt.savefig("comparacao_weight_tying.png", dpi=300)
plt.close()

tempo_total = (time.time() - inicio_global) / 60
print(f"\n[SUCESSO] Execução concluída em {tempo_total:.1f} minutos!")
