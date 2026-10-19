# Seminário I — Questão 15: The Shape of an Embedding Table (`BERT-cpu`)

**Autor:** Kenneth Yan Santana Oliveira  
**Disciplina:** Redes Neurais Artificiais (2026.2) — PPCOMP / UFAPE  
**Docente:** Prof. Dr. Luis Filipe  

---

## 1. The Question (A Pergunta de Pesquisa)

No exercício base (`q06_learn_embedding.ipynb`), o modelo Word2Vec (*Skip-Gram with Negative Sampling* — SGNS) é treinado sobre o corpus do livro *Flatland* fixando arbitrariamente a dimensão do embedding em $D = 32$, o número de amostras negativas em $K = 5$ e $100$ épocas. Contudo, cada um desses hiperparâmetros atua diretamente como um multiplicador do custo aritmético (FLOPs).

A **Questão 15** investiga experimentalmente duas perguntas abertas fundamentais:
1. **Qual é a configuração ótima em custo computacional (FLOP-optimal) de $(D, K)$ para maximizar a separação semântica (medida pelo coeficiente de Silhueta) neste corpus?**
2. **Amarrar as tabelas de entrada e saída (*Weight Tying*, `tie_tables=True`) reduz os parâmetros livres pela metade "de graça" (sem perda de qualidade semântica)?**

Adicionalmente, atendendo à análise de limites assintóticos de compressão, investigamos o comportamento de **subajuste (*underfitting*) em dimensões ultra-restritas ($D \in \{2, 4\}$)** frente ao **sobreajuste (*overfitting*) em dimensões elevadas ($D \in \{32, 64\}$)**.

---

## 2. What the Literature Says (Fundamentação Teórica e Trabalhos Relacionados)

* **Mikolov et al. (2013) — *Distributed Representations of Words and Phrases*:** Formula o objetivo SGNS substituindo a Softmax sobre todo o vocabulário por problemas binários logísticos com $K$ amostras de ruído sorteadas segundo a distribuição unigrama suavizada $P(w) \propto f(w)^{3/4}$.
* **Levy & Goldberg (2014) — *Neural Word Embedding as Implicit Matrix Factorization*:** Demonstra matematicamente que o treinamento SGNS é equivalente a fatorar implicitamente uma matriz de Informação Mútua Pontual (*Pointwise Mutual Information* — PMI) deslocada por $-\log(K)$:
  $$\mathbf{v}_w^\top \mathbf{u}_c = \text{PMI}(w, c) - \log(K)$$
  Assim, a escolha de $D$ deixa de ser um ajuste cego e passa a representar uma aproximação de baixo posto (*low-rank*) dependente do espectro de valores singulares da matriz do corpus.
* **Yin & Shen (2018) — *On the Dimensionality of Word Embedding*:** Estabelece formalmente o *trade-off* de **Viés-Variância** na escolha da dimensão $D$. Em um corpus finito e ruidoso:
  * **Dimensões muito baixas ($D \to 2$)** sofrem de **alto viés estrutural (*underfitting*)**: truncam autovalores de sinal semântico relevante e não possuem graus de liberdade geométricos suficientes.
  * **Dimensões muito altas ($D \to 64$) em corpora pequenos** sofrem de **alta variância (*overfitting* latente)**: o excesso de parâmetros livres passa a reconstruir o ruído estocástico das coocorrências esparsas, degradando a geometria dos agrupamentos.
* **Press & Wolf (2017) — *Using the Output Embedding to Improve Language Models*:** Propõe o compartilhamento de pesos (*Weight Tying*, $\mathbf{U} = \mathbf{V}$) entre tabelas de entrada e saída. Embora eficaz em grandes modelos de linguagem (como Transformers e BERT), no objetivo puro do SGNS essa restrição força cada palavra a compartilhar o mesmo vetor como centro e contexto.

---

## 3. Hypothesis (Hipótese Científica)

1. **Curva em U Invertido (Viés-Variância em $D$):** Como o *Flatland* é um corpus de domínio fechado e vocabulário compacto ($V = 886$ palavras ativas e $142.848$ pares de treino), a qualidade semântica não crescerá monotonicamente com $D$. Esperamos observar **subajuste severo em $D=2$**, atingir um **ponto ótimo (FLOP-optimal) em dimensões compactas ($D \in \{4, 8\}$)** e sofrer **degradação progressiva por sobreajuste em $D \in \{32, 64\}$**.
2. **Impacto de $K$ em Vocabulário Pequeno:** Valores menores de amostragem negativa ($K=2$) serão mais eficientes em FLOPs e evitarão falsos negativos (colisões semânticas em um vocabulário de apenas 886 palavras) e deslocamentos excessivos de $-\log(K)$.
3. **Custo do *Weight Tying* no SGNS:** Ao ativar `tie_tables=True`, a simetrização forçada ($\mathbf{u}_w = \mathbf{v}_w$) degradará o coeficiente de Silhueta, demonstrando que a redução de parâmetros não ocorre sem perda de qualidade no Word2Vec puro.

---

## 4. What Was Implemented & Protocol (Metodologia e Protocolo Experimental)

Utilizamos o instrumental nativo `research/benchmark.py` (`run_sgns`, `summarize` e `pareto`) acoplado ao script automatizado **`rodar_pesquisa.py`**:

* **Grid Bidimensional Principal:** $D \in \{2, 4, 8, 16, 32, 64\} \times K \in \{2, 5, 10\}$ (18 configurações).
* **Experimento de *Weight Tying*:** Comparação direta entre tabelas separadas (`tie_tables=False`) e tabelas amarradas (`tie_tables=True`) para $D \in \{8, 32, 64\}$ e $K \in \{2, 5\}$.
* **Hiperparâmetros Fixos:** `epochs = 100`, `batch = 2048`, `lr = 0.02` (Adam), `window = 3`, `min_count = 5`.
* **Rigor Estatístico:** Todas as configurações foram executadas com **$n = 3$ sementes independentes (`seeds = [0, 1, 2]`)**, reportando **Média ± Desvio Padrão** para o Coeficiente de Silhueta (distância de cosseno sobre 18 palavras divididas nos grupos `shapes`, `people` e `space`) e contagem exata de **GFLOPs de treinamento**.
* **Sanity Check Qualitativo:** Extração do detalhamento de Silhueta por palavra/cluster e inspeção dos 5 vizinhos mais próximos (*Nearest Neighbours*) por similaridade de cosseno.

---

## 5. Results (Resultados Experimentais)

### 5.1. Varredura Principal ($D \times K$) vs. Baseline (`D32_K5`)

```text
configuration                 n             score   train GFLOP   infer MFLOP     params    s/run
-------------------------------------------------------------------------------------------------
D2_K2                         3   -0.0061 ± 0.0169          0.49           0.0      3,544     27.0
D2_K5                         3   -0.0402 ± 0.0082          1.01           0.0      3,544     43.2
D2_K10                        3   -0.0407 ± 0.0127          1.86           0.0      3,544     70.5
D4_K2                         3    0.3728 ± 0.0378          0.68           0.0      7,088     30.9
D4_K5                         3    0.3012 ± 0.1055          1.38           0.0      7,088     50.3
D4_K10                        3    0.3316 ± 0.0493          2.54           0.0      7,088     82.2
D8_K2                         3    0.3134 ± 0.0520          1.05           0.0     14,176     39.6
D8_K5                         3    0.3025 ± 0.0365          2.12           0.0     14,176     63.4
D8_K10                        3    0.3118 ± 0.0519          3.90           0.0     14,176    104.1
D16_K2                        3    0.2373 ± 0.0227          1.79           0.0     28,352     56.2
D16_K5                        3    0.2810 ± 0.0150          3.60           0.0     28,352     91.7
D16_K10                       3    0.2902 ± 0.0297          6.62           0.0     28,352    151.5
D32_K2                        3    0.1934 ± 0.0210          3.28           0.0     56,704     87.0
D32_K5 (baseline)             3    0.1636 ± 0.0215          6.57           0.0     56,704    143.5
D32_K10                       3    0.1390 ± 0.0324         12.06           0.0     56,704    239.5
D64_K2                        3    0.0782 ± 0.0198          6.25           0.0    113,408    152.6
D64_K5                        3    0.0820 ± 0.0068         12.51           0.0    113,408    251.5
D64_K10                       3    0.0933 ± 0.0255         22.95           0.0    113,408    421.7
