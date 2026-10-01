Section 1: The Question (A Pergunta) 
- Objetivo: Avaliar empiricamente o impacto da variação conjunta da dimensionalidade latente (\[D\]) e da amostragem negativa (\[K\]) na qualidade dos embeddings gerados pelo algoritmo Skip-Gram com Amostragem Negativa (SGNS).
- Métrica Central: Desempenho semântico avaliado através do coeficiente Silhouette de 3 grupos de controle vs custo computacional bruto computado estritamente em operações de ponto flutuante (FLOPs).

Section 2: What the Literature Says (O que diz a Literatura)
- Abordagem Teórica (Levy & Goldberg, 2014): Demonstração formal de que o algoritmo SGNS realiza implicitamente a fatoração de uma matriz PMI deslocada pelo fator \(\log(K)\).
- Balanço Viés-Variância (Yin & Shen, 2018): Previsão matemática de que em corpora compactos e de domínio restrito (como o Flatland), o excesso de dimensões gera saturação assintótica e codificação de ruído amostral.

Section 3: Hypothesis (Hipótese Científica) 
- Enunciado Falseável: "Em conjuntos de dados de escopo reduzido, o estreitamento dimensional forçado por tabelas compactas (gargalo de \(D=8\)) resultará em clusters semânticos mais densos e coesos, superando a acurácia Silhouette de baselines de dimensões mais altas, enquanto o incremento de amostras negativas (\[K\]) atuará como estabilizador de variância."

Section 4: What You Implemented (O que foi implementado) 
- Infraestrutura: Script de automação e varredura bidimensional unificada (rodar_pesquisa.py) acoplado diretamente ao módulo instrumental original do repositório (research/benchmark.py).
- Modificações: Integração direta com a API nativa de contagem global de FLOPs e funções de agregação estatística para exportação de dados analíticos.

Section 5: Protocol (Protocolo Experimental) 
- Variáveis Independentes (Knobs): Dimensões \(D \in \{8, 32, 64\}\) cruzadas com amostras de ruído \(K \in \{2, 5, 10\}\).
- Variáveis Dependentes (Métricas): Média aritmética de Silhouette Score, contagem exata de GFLOPs de treinamento e tempo de execução em segundos.
- Elementos Fixos: Limite estrito de 100 épocas de otimização por execução.
- Rigor Estatístico: Avaliação conduzida sobre 3 sementes distintas (seeds 0, 1 e 2) para eliminação de anedotas numéricas.

Section 6: Results (Resultados) 
- Desempenho Semântico: A menor configuração (\(D8\_K2\)) consolidou a maior acurácia (0.3134), demonstrando compressão de sinal ideal para o domínio restrito.
- Custo Computacional: A expansão paramétrica para \(D64\_K10\) gerou ineficiência severa, escalando o custo para 22.95 GFLOPs enquanto degradou o score Silhouette para 0.0933.
- Comportamento de Pareto: O gráfico de fronteira ilustra uma trajetória negativa, onde o investimento em maiores orçamentos computacionais redunda em saturação estrutural do espaço latente. 

Section 7: Threats to Validity (Ameaças à Validade) 
- Aproximação do Contador: O módulo engine.py contabiliza com precisão estrita os produtos matriciais pesados (matmuls), mas realiza aproximações lineares nas operações elementares secundárias.
- Escala do Dataset: A dinâmica observada é correlata ao tamanho amostral concentrado do Flatland Corpus, podendo apresentar comportamento inverso (curvas ascendentes) quando transposta para corpora de escala industrial (ex: ImageNet ou LLMs de bilhões de parâmetros).

Section 8: What Remains Open (Trabalho Futuro) 
- Próximos Passos: Investigar o impacto da amarração de tabelas de entrada e saída (weight tying) para avaliar se a redução de 50% dos parâmetros livres estabiliza a dispersão de performance nas dimensões intermediárias (\(D=32\)). 
