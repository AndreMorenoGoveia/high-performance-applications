# Planejamento Completo do Estudo e Artigo Científico (Versão Revisada — Rigor Técnico & Metodológico)

## 1. Identificação do Projeto e Posicionamento Honesto de Novidade

- **Título Provisório (PT-BR):** *BitMamba-CV: Co-Design Hardware-Algoritmo de Modelos de Espaço de Estados Mamba-2 Ternários (1.58-bit) com Discretização CPoT em FPGA de Borda*
- **Working Title (EN):** *BitMamba-CV: Hardware-Algorithm Co-Design of 1.58-Bit Ternary Mamba-2 State Space Models with Complementary Power-of-Two Discretization on Edge FPGA*
- **Arquitetura de Modelo Escolhida:** **Mamba-2 (Structured State Space Duality — SSD)** com transição $A_t$ **escalar por cabeça (*scalar per-head*)**, o que reduz a complexidade do circuito de discretização $\exp(\Delta_t A)$ de $O(D \times N)$ (no Mamba-1) para apenas $O(H)$ cabeças ($H \in \{4, 8\}$).
- **Plataforma Alvo:** **Intel Cyclone V SoC** (28nm — ex: DE10-Nano `5CSEBA6U23I7` / DE1-SoC `5CSEMA5F31C6`, integrando ARM Cortex-A9 Dual-Core HPS + malha FPGA com ALMs fracionáveis de 6 entradas, blocos M10K BRAM e DSPs de precisão variável) via **Intel Quartus Prime**.
- **Veículos Alvo:** **SSCAD** / **WPerformance** (formato SBC, até 12 páginas) ou **SBAC-PAD / FPL / FCCM**.

### 1.1 Onde Está (e Onde NÃO Está) a Nossa Novidade Científica
Para blindar o artigo contra revisores especialistas em arquitetura de FPGAs e quantização de LLMs, fazemos uma atribuição explícita e honesta do que já existe na literatura recente:
- **O que JÁ existe (e citamos explicitamente):**
  - Empacotamento de 5 trits em 1 byte ($1.60\text{ bits/peso}$, *Radix-3*) e árvores somadoras ternárias ou *Table-Lookup* em FPGA já foram propostos para **Transformers** em *TerEffic* ([arXiv:2502.16473](https://arxiv.org/abs/2502.16473)), *TeLLMe* ([arXiv:2504.16266](https://arxiv.org/abs/2504.16266)) e em *kernels* de CPU como `TQ1_0`/`TQ2_0` do `llama.cpp` e *T-MAC* ([arXiv:2407.00088](https://arxiv.org/abs/2407.00088)).
  - Aceleração de Mamba em FPGA com estado recorrente $O(1)$ *on-chip* e quantização em potência de dois (PoT) convencional ($2^{-k}$) já foi proposta para modelos **W4A4 e W8A8** em FPGAs de grande porte em *LightMamba* ([arXiv:2502.15260](https://arxiv.org/abs/2502.15260)), *FastMamba* ([arXiv:2505.18975](https://arxiv.org/abs/2505.18975)) e *MARCA* (ICCAD 2024).
- **O que é genuinamente INÉDITO no nosso artigo (Nossas 4 Contribuições Reais):**
  1. **Primeiro Co-Design Fim-a-Fim (QAT + RTL) de Mamba-2 Ternário ($1.58\text{-bit}$) com `H-BitLinear` em FPGA de Borda:** Unificamos o treinamento *Quantization-Aware Training* (QAT) com Transformada de Hadamard *in-graph* (inspirado no *BitNet v2*, [arXiv:2504.18415](https://arxiv.org/abs/2504.18415), preservando os pesos estritamente ternários $\tilde{W} \in \{-1,0,+1\}$) à recorrência $O(1)$ do **Mamba-2**, eliminando o gargalo de KV-Cache $O(L)$ dos aceleradores BitNet anteriores (*TerEffic* / *TeLLMe*).
  2. **Discretização *Complementary Power-of-Two* (CPoT: $\bar{A}_t \in \{2^{-k}, 1 - 2^{-k}\}$) para Preservação de Memória Longa:** Provamos matemática e experimentalmente que a quantização PoT convencional ($2^{-k} \in \{1, 0.5, 0.25, \dots\}$) usada em *LightMamba*/*FastMamba* destrói os canais de memória longa do SSM onde $\bar{A}_t \in (0.90, 0.999)$. Propomos a discretização **CPoT** ($1 - 2^{-k}$), que concentra resolução próxima de $1.0$ (ex: $0.96875, 0.99218, 0.99902$) e executa em hardware sem multiplicadores via $h_{t-1} - (h_{t-1} \gg k)$.
  3. **Exploração Microarquitetural nos ALMs de 6 Entradas da Cyclone V (Árvore Somadora vs. *Zero-Skipping* vs. *Table-Lookup* vs. Baseline INT8 com DSP):** Como cerca de $35\%\text{--}45\%$ dos pesos em camadas `BitLinear` treinadas são iguais a zero ($w=0$) e os DSPs já existem gratuitamente no silício da FPGA, realizamos uma exploração justa (*same-chip DSE*) comparando três microarquiteturas ternárias em ALMs contra um baseline Mamba-2 INT8 nos DSPs da própria Cyclone V.
  4. **Caracterização Honesta de Desempenho (*Roofline* Dual-Regime) e Validação em Placa Real (HPS+FPGA):** Avaliamos o sistema ponta a ponta na Cyclone V SoC (com transferência real via barramento AXI e medição física de potência), demonstrando a transição entre o regime *Compute-Bound* (modelos compactos de borda 100% em M10K) e o regime *Memory-Bandwidth-Bound* (modelos maiores paginados via DDR3 em *batch=1*, onde o empacotamento *Radix-3* entrega ganho proporcional à redução de bytes trafegados).

---

## 2. Protocolo de Busca Sistemática e Literatura Verificada (2024–2026)

### 2.1 Registro da Busca Sistemática (Para inclusão na Seção 3 do Artigo)
- **Bases consultadas:** arXiv (`cs.AR`, `cs.LG`, `eess.SP`), IEEE Xplore, ACM Digital Library e ACL Anthology.
- **Strings de busca registradas (Data-base: Outubro/2026):**
  - `("Mamba" OR "State Space Model" OR "SSM") AND ("ternary" OR "1.58-bit" OR "BitNet" OR "1-bit") AND ("FPGA" OR "hardware" OR "accelerator")`
  - `("BitMamba" OR "Bi-Mamba" OR "Slender-Mamba" OR "Quamba") AND ("FPGA" OR "RTL" OR "systolic")`
- **Resultado da triagem:** Existem trabalhos algorítmicos de Mamba 1-bit/1.58-bit em GPU/CPU (*Bi-Mamba*, *Slender-Mamba*, *BitMamba-2*, *Ternary Mamba*) e aceleradores FPGA para Mamba em 4/8 bits (*MARCA*, *LightMamba*, *FastMamba*, *SpecMamba*, *eMamba*, *ViM-Q*) ou para BitNet Transformer (*TerEffic*, *TeLLMe*). A síntese co-projetada de **Mamba-2 Ternário ($1.58\text{-bit}$) com QAT Hadamard + CPoT SSM em FPGA** permanece um problema em aberto.

### 2.2 Tabela Bibliográfica Auditada (Referências Separadas e Verificadas)

| Trabalho | Autores / Ano / Veículo / ID | Arquitetura Alvo | Pesos / Ativações | Tratamento do SSM ($\bar{A}_t, \bar{B}_t, C_t$) | Plataforma Avaliada | Relação Exata com o *BitMamba-CV* |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **BitNet b1.58** | Ma et al., 2024 ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) | Transformer | $1.58\text{-bit}$ / INT8 | N/A (Atenção + KV-Cache $O(L)$) | GPU / CPU | Base da quantização ternária `RoundClip` e `absmax`. |
| **BitNet v2** | Wang et al., 2025 ([arXiv:2504.18415](https://arxiv.org/abs/2504.18415)) | Transformer | $1.58\text{-bit}$ / INT4-8 | N/A (Usa **H-BitLinear** com Hadamard no QAT) | GPU | Resolve o conflito entre FWHT e pesos ternários treinando $W$ após $\mathbf{H}$ no QAT. |
| **Bi-Mamba** | Tang et al., Nov 2024 ([arXiv:2411.11843](https://arxiv.org/abs/2411.11843), TMLR) | Mamba-1 | $1\text{-bit}$ (Binário) / FP16 | Recorrência em precisão alta | GPU (PyTorch) | Focado em pesos binários $\{-1,+1\}$ com destilação; sem hardware dedicado. |
| **Slender-Mamba** | Yu et al., Jan 2025 (COLING 2025) | Mamba-2 | $1.58\text{-bit}$ (Ternário) | SSM mantido em precisão padrão | GPU | Prova que Mamba-2 ternário funciona *head-to-toe*, mas apenas em software. |
| **Ternary Mamba** | Ganesaraja et al., Jun 2026 ([arXiv:2606.18114](https://arxiv.org/abs/2606.18114)) | Mamba-2 (1.3B) | W1.58A16 (Grouped QAT) | SSM em FP16 | GPU | Mostra que PTQ falha em Mamba ternário e exige QAT; não quantiza ativações nem SSM para FPGA. |
| **Quamba / Quamba2** | Chiang et al., 2024/2025 ([arXiv:2410.13889](https://arxiv.org/abs/2410.13889), [2503.22879](https://arxiv.org/abs/2503.22879)) | Mamba-1/2 | W8A8 / W4A8 | Hadamard no PTQ (absorvida em pesos INT4/8) | GPU (Orin / A5000) | A absorção $W \mathbf{H}^T$ do Quamba destrói pesos ternários; por isso usamos QAT estilo *BitNet v2*. |
| **MARCA** | Li et al., Out 2024 (IEEE/ACM ICCAD 2024) | Mamba-1 | FP16 / INT8 | PEs reconfiguráveis com árvore de redução | ASIC / Simulação | Usa multiplicadores densos; não aborda pesos ternários nem FPGAs de baixo custo. |
| **LightMamba** | Wei et al., Fev 2025 ([arXiv:2502.15260](https://arxiv.org/abs/2502.15260)) | Mamba-2 | W4A4 | Rotação + PoT convencional ($2^{-k}$) no SSM | Xilinx Alveo U280 & Versal VCK190 | Depende de DSPs para projeções 4-bit; PoT simples ($2^{-k}$) perde resolução perto de $\bar{A}_t \approx 1$. |
| **FastMamba** | Maio 2025 ([arXiv:2505.18975](https://arxiv.org/abs/2505.18975)) | Mamba-2 | W8A8 (Linear) + PoT | Hadamard + PoT no SSM/Conv + PWL SiLU | Xilinx Virtex-7 VC709 | Mantém $90\%$ dos pesos (`in_proj`/`out_proj`) em 8 bits (5x maior que *Radix-3*). |
| **TerEffic** | Yin et al., Fev 2025 ([arXiv:2502.16473](https://arxiv.org/abs/2502.16473)) | Transformer (BitNet) | $1.58\text{-bit}$ ($1.6\text{-bit}$ packed) / INT8 | N/A (Gargalo de KV-Cache $O(L)$) | Xilinx Alveo U280 (HBM) | Introduz compressão $1.6\text{-bit}$ e somadores ternários em FPGA, mas preso ao KV-Cache de Transformers. |
| **TeLLMe** | Qiao et al., Abr 2025 ([arXiv:2504.16266](https://arxiv.org/abs/2504.16266)) | Transformer (BitNet) | $1.58\text{-bit}$ / INT8 | N/A (Atenção fundida + KV-Cache em DDR) | AMD Kria KV260 | Usa *Table-Lookup MatMul* (TLMM) em FPGA de borda para Transformers; sofre com KV-Cache em $L$ longo. |
| **BitMamba-CV (Nosso)** | **Proposto (2026)** | **Mamba-2 (SSD)** | **$1.58\text{-bit}$ (Radix-3) / INT8** | **CPoT ($\{2^{-k}, 1-2^{-k}\}$) no $\bar{A}_t$ + DSP INT8 em $\bar{B}_t x_t, C_t h_t$** | **Intel Cyclone V SoC (DE10-Nano / DE1-SoC)** | **Primeiro co-design QAT `H-BitLinear` + Mamba-2 CPoT em FPGA, com baseline INT8 no mesmo chip.** |

---

## 3. Correção dos 5 Pontos Técnicos Críticos (Formulação Matemática Exata)

### 3.1 Solução do Ponto 1: Como usar a FWHT sem quebrar os pesos ternários (`H-BitLinear` via QAT)
- **O Problema Matemático:** Em *Post-Training Quantization* (como no *Quamba* e *QuaRot*), insere-se a identidade $\mathbf{H}^T \mathbf{H} = \mathbf{I}$ entre a ativação $\mathbf{x}$ e o peso $\mathbf{W}$:
  $$\mathbf{y} = \mathbf{W} \mathbf{x} = (\mathbf{W} \mathbf{H}^T) (\mathbf{H} \mathbf{x})$$
  Se $\mathbf{W} \in \{-1, 0, +1\}^{M \times D}$, a matriz rotacionada $\mathbf{W}' = \mathbf{W} \mathbf{H}^T$ **deixa de ser ternária** (passa a ter múltiplos níveis inteiros/reais), inviabilizando o motor sem multiplicadores!
- **Nossa Solução (`H-BitLinear` em QAT):** Adotamos a formulação do *BitNet v2* ([arXiv:2504.18415](https://arxiv.org/abs/2504.18415)) durante o **Treinamento Ciente de Quantização (QAT)** do nosso modelo Mamba-2:
  1. A Transformada Rápida de Walsh-Hadamard em blocos $\mathbf{H}_B$ ($B = 16$ ou $32$, com escalonamento exato por *shift* $\frac{1}{\sqrt{B}} = 2^{-k}$ para $B \in \{16, 64\}$) faz parte **direta do grafo de *forward pass*** antes da quantização da ativação:
     $$\tilde{\mathbf{x}} = \text{Quant}_{\text{INT8}}\left( 2^{-k} \mathbf{H}_B \mathbf{x} \right)$$
  2. A matriz de pesos $\mathbf{W}$ é treinada do zero via *Straight-Through Estimator* (STE) já no espaço transformado de Hadamard:
     $$\tilde{\mathbf{W}} = \text{RoundClip}\left( \frac{\mathbf{W}}{\gamma_W}, -1, +1 \right) \in \{-1, 0, +1\}^{M \times D}, \quad \gamma_W = \frac{1}{MD}\sum_{i,j} |W_{ij}|$$
  3. Assim, em tempo de inferência na Cyclone V, a FWHT suaviza os *outliers* de $\mathbf{x}$ usando apenas somas/subtrações e *bit-shift*, e os pesos $\tilde{\mathbf{W}}$ armazenados na M10K **continuam 100% ternários $\{-1, 0, +1\}$**!

### 3.2 Solução do Ponto 2: Transparência Total no Orçamento de Multiplicadores (Onde entram os DSPs vs. ALMs)
Não venderemos a ilusão de um chip "100% sem multiplicadores". O artigo trará uma seção específica (**Seção 4.5 — Alocação Heterogênea ALM vs. DSP**) e um gráfico de barras empilhadas/pizza detalhando onde cada recurso da Cyclone V é empregado:

| Sub-Bloco do Mamba-2 | Operação Matemática | % dos Parâmetros | % dos MACs/OPs | Recurso Usado na Cyclone V | Usa Bloco DSP? |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **1. Projeções Lineares (`in_proj`, `out_proj`)** | $\mathbf{y} = \gamma \cdot (\tilde{\mathbf{W}}_{\text{tern}} \cdot \tilde{\mathbf{x}}_{\text{INT8}})$ | **~92%** | **~86%** | **ALMs** (Soma/Sub condicional ou LUT) + **1 DSP** no final para o fator de escala $\gamma$ | **0 DSPs no produto interno** (1 DSP compartilhado no reescalonamento) |
| **2. Rotação de Outliers (`FWHT` $B=16$)** | $\tilde{\mathbf{x}} = (\mathbf{H}_{16} \mathbf{x}) \gg 2$ | 0% | ~3% | **ALMs** (4 estágios de soma/subtração em borboleta + *wiring shift*) | **0 DSPs** |
| **3. Convolução Curta (`Conv1D` $k=4$)** | $\sum_{c=0}^{3} w_c \cdot x_{t-c}$ (*Depthwise*) | <1% | ~2% | **DSPs** (Modo dual $9\times 9$ ou $18\times 18$) | **Sim (DSPs)** |
| **4. Discretização Recorrente ($\bar{A}_t \cdot h_{t-1}$)** | $\bar{A}_t = \exp(-\Delta_t A_h) \in \{2^{-k}, 1-2^{-k}\}$ | <1% | ~2% | **ALMs** (*Barrel Shifter* + Subtrator CPoT) | **0 DSPs** |
| **5. Injeção e Leitura de Estado ($\bar{B}_t x_t$, $C_t h_t$) e Gate ($y_t \odot \text{SiLU}(z_t)$)** | Produtos dinâmicos $\text{INT8} \times \text{INT8/16}$ e interpolação PWL de `SiLU`/`Softplus` | ~7% | ~6% | **DSPs** + **M10K** (Tabelas PWL $a \cdot x + b$) | **Sim (DSPs)** |
| **6. Normalização (`RMSNorm`) e Quantização Dinâmica (`absmax`)** | $\text{max}_i |x_i|$, LUT recíproca $1/\sigma$ e produto por canal | <1% | ~1% | **ALMs** (Árvore de comparadores `max`) + **M10K** (`rsqrt` / recíproco LUT) + **DSP** | **Sim (1–2 DSPs compartilhados)** |

> **Argumento Arquitetural Forte:** Em vez de deixar os 87–112 blocos DSP da Cyclone V ociosos (o que seria desperdício de silício já pago), nós **desalojamos as projeções lineares (86% das operações) para os ALMs via aritmética ternária**, permitindo que os **blocos DSP fiquem 100% dedicados às operações não-lineares, convolução `Conv1D` e produtos dinâmicos $\bar{B}_t x_t$ e $C_t h_t$ do SSM** sem contenção nem multiplexação temporal pesada!

### 3.3 Solução do Ponto 3: Discretização *Complementary Power-of-Two* (CPoT) para $\bar{A}_t$ perto de $1.0$
- **Por que o PoT padrão falha em $L$ longo:** No Mamba, a memória de longo prazo depende de autovalores discretizados $\bar{A}_t = \exp(-\Delta_t A) \rightarrow 1^-$ (tipicamente $0.95 \le \bar{A}_t \le 0.999$). Na quantização PoT padrão ($\bar{A}_t = 2^{-k}$), os únicos níveis disponíveis são $\{1.0, 0.5, 0.25, 0.125, \dots\}$. Qualquer canal com $\bar{A}_t \in (0.75, 0.999)$ é arredondado para $1.0$ (causando explosão/não-esquecimento do estado $h_t$) ou para $0.5$ (apagando a memória em apenas 5 passos de tempo!).
- **Nossa Formulação CPoT (Bimodal Power-of-Two):**
  Pela expansão de Taylor para passos $\Delta_t A$ pequenos, $\exp(-\Delta_t A) \approx 1 - \Delta_t A$. Portanto, quantizamos $\bar{A}_t$ usando 1 bit de modo (`m`) e um deslocamento $k \in \{1, \dots, 12\}$:
  $$\bar{A}_t^{\text{CPoT}}(m, k) = \begin{cases} 2^{-k}, & \text{se } m = 0 \quad \text{(regime de decaimento rápido: } \bar{A}_t \le 0.5\text{)} \\ 1 - 2^{-k}, & \text{se } m = 1 \quad \text{(regime de memória longa: } 0.5 < \bar{A}_t < 1\text{)} \end{cases}$$
- **Mapeamento em Hardware (Zero Multiplicadores):**
  $$h_t = \bar{A}_t^{\text{CPoT}} \cdot h_{t-1} + \bar{B}_t x_t = \begin{cases} (h_{t-1} \gg k) + \bar{B}_t x_t, & \text{se } m = 0 \\ h_{t-1} - (h_{t-1} \gg k) + \bar{B}_t x_t, & \text{se } m = 1 \end{cases}$$
  Com apenas 1 *barrel shifter* e 1 subtrator, obtemos os níveis $\{0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875, 0.99609, 0.99804, 0.99902\}$ exatamente na zona crítica de memória longa! (Também avaliaremos a variante de 2 shifts $1 - 2^{-k_1} - 2^{-k_2}$ na tabela de ablação).

### 3.4 Solução do Ponto 4: Análise *Roofline* Honesta nos Dois Regimes de Memória
Na Seção 5, apresentaremos o modelo *Roofline* separado de forma transparente em dois cenários reais da Cyclone V SoC:
1. **Configuração A (*On-Chip Edge Mamba-2* — Pesos Residentes nas M10K BRAMs, até ~550 KB = ~2.8M parâmetros em $1.6\text{-bit}$):**
   - Todos os pesos e o estado $O(1)$ cabem dentro dos blocos M10K da Cyclone V (largura de banda interna agregada $> 60\text{ GB/s}$ a $150\text{ MHz}$). Neste regime, a aplicação opera no platô **Compute-Bound** da FPGA mesmo em `batch=1`, entregando latência determinística na casa de poucos microssegundos para classificação de áudio/ECG/sensores.
2. **Configuração B (*Off-Chip Streaming SLM* — Pesos na DDR3 via Barramento AXI HPS-to-FPGA, banda prática $\approx 1.8\text{--}3.2\text{ GB/s}$):**
   - Em `batch=1`, cada token precisa ler todos os pesos da DDR3, resultando em intensidade operacional baixa ($\approx 1.25\text{ OPs/bit} = 10\text{ OPs/Byte}$). Portanto, a Configuração B em `batch=1` é **estritamente Memory-Bandwidth Bound** (fica na rampa inclinada do Roofline!).
   - **O que o Roofline prova honestamente na Config B:** Na rampa limitada por memória, o *throughput* ($\text{Tokens/s}$) é diretamente proporcional ao inverso do tamanho do peso em bits! Assim, comprimir os pesos de INT8 ($8\text{ bits}$) para *Radix-3* ($1.6\text{ bits}$) proporciona um **ganho real de $5\times$ em Tokens/s**, e comprimir de 2 bits para $1.6\text{ bits}$ proporciona **+25% de Tokens/s diretos na banda DDR3**! Já no estágio de *Prefill* (processando múltiplos tokens em paralelo, ex: chunk de 16 ou 32 tokens), a intensidade aritmética multiplica por $16\times\text{--}32\times$, deslocando a execução da rampa de memória para o teto de computação.

### 3.5 Solução do Ponto 5: Pipeline Completo de Treinamento QAT do Zero (`sw/train_qat_bitmamba2.py`)
Como não existem *checkpoints* pré-treinados de Mamba-2 ternário para tarefas de borda com `H-BitLinear` e `CPoT`, a **Fase 1** do projeto dedica-se a treinar do zero em PyTorch modelos **Mamba-2 compactos ($D \in \{64, 128\}$, $N=16$, $2\text{ a }4$ camadas)** com QAT + STE em 3 benchmarks reais com comprimentos de sequência crescentes:
1. **ECG5000 / MIT-BIH Arrhythmia ($L = 140\text{ a }360$):** Detecção médica de arritmia em borda (convergência rápida em minutos para depuração de ablação).
2. **Google Speech Commands v2 ($L = 1600$ amostras / espectrograma streaming):** Reconhecimento de comandos de voz em tempo real.
3. **Sequential MNIST / Permuted Sequential MNIST ($L = 784$) e Copy/Induction Memory Task ($L = 512 \text{ até } 16.384$):** Benchmark clássico do NeurIPS para provar experimentalmente que o **CPoT ($1 - 2^{-k}$)** preserva a acurácia em sequências longas onde o PoT tradicional ($2^{-k}$) colapsa!

---

## 4. Metodologia Experimental de Alto Nível (Soluções dos Pontos 6 a 12)

### 4.1 Baseline Justo no Mesmo Chip (Ponto 6)
Para isolar o ganho real da arquitetura proposta sem confundir tecnologia de litografia ou placa, sintetizaremos **no mesmo FPGA Intel Cyclone V (`5CSEBA6U23I7` / `5CSEMA5F31C6`)** sob as mesmas restrições de *timing* e grau de paralelismo $P$:
1. **`Mamba2-INT8-DSP` (Baseline Mesmo-Chip):** Bloco Mamba-2 quantizado em W8A8 onde o produto interno linear usa os multiplicadores físicos dos blocos DSP da Cyclone V.
2. **`BitMamba2-2bit-Adder` (Baseline Ternário Simples):** Pesos ternários em 2 bits com árvore somadora direta nos ALMs.
3. **`BitMamba2-Radix3-Adder` vs. `BitMamba2-Radix3-TLMM` vs. `BitMamba2-ZeroSkip` (Ponto 12):**
   - Comparamos a **Árvore Somadora Ternária** nos ALMs contra a abordagem de **Busca em Tabela (*Table-Lookup MatMul* — TLMM estilo *T-MAC*/*TeLLMe*, agrupando 3 ou 4 pesos ternários na M10K/ALMs)** e um esquema de **Zero-Skipping Estruturado** (explorando os ~40% de pesos zero das camadas `BitLinear`).
- **Métrica de Eficiência de Área Normalizada:** Além de reportar ALMs, M10K e DSPs separadamente, calcularemos a área equivalente normalizada ($\text{ALMs}_{\text{eq}} = \text{ALMs} + \lambda_{\text{DSP}} \cdot \text{DSPs}$, onde $\lambda_{\text{DSP}} \approx 120\text{ ALMs}$ equivale ao custo lógico de 2 multiplicadores $18\times 18$ na Cyclone V) e traçaremos a **Fronteira de Pareto ($\text{Tokens/s}$ por $\text{ALM}_{\text{eq}}$)**.

### 4.2 Tabela Obrigatória de Estudo de Ablação (Ponto 7)
O artigo conterá uma tabela dedicada avaliando o impacto incremental de cada componente (média $\pm$ desvio padrão em 3 *seeds*):

| Variante do Modelo / Hardware | Pesos | Codificação M10K | Motor Linear | Rotação QAT | Discretização $\bar{A}_t$ | `SiLU` | ALMs | Blocos M10K | DSPs | Latência ($\mu\text{s}$) | SQNR (dB) | Acurácia (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0: Baseline FP32 (Software)** | FP32 | — | FP32 | Não | Exata ($\exp$) | Exata | — | — | — | — | $\infty$ | Ref. |
| **A1: Baseline FPGA INT8-DSP** | INT8 | 8-bit | DSP MAC | Não | Fixo 16-bit | LUT | Medido | Medido | Alto | Medido | Medido | Medido |
| **A2: Ternário Ingênuo (Sem FWHT)** | $1.58\text{b}$ | 2-bit | Adder Tree | **Não** | Fixo 16-bit | LUT | Medido | Medido | Baixo | Medido | Cai | Cai |
| **A3: + QAT `H-BitLinear` (FWHT)** | $1.58\text{b}$ | 2-bit | Adder Tree | **Sim** | Fixo 16-bit | LUT | Medido | Medido | Baixo | Medido | Sobe | Recupera |
| **A4: + Empacotamento *Radix-3*** | $1.58\text{b}$ | **$1.6\text{-bit}$** | Adder Tree | Sim | Fixo 16-bit | LUT | +Decoder | **-20%** | Baixo | Medido | Igual | Igual |
| **A5: + Motor `TLMM` / `Zero-Skip`** | $1.58\text{b}$ | $1.6\text{-bit}$ | **TLMM/Skip** | Sim | Fixo 16-bit | LUT | Medido | Medido | Baixo | Menor | Igual | Igual |
| **A6: + SSM com PoT Padrão ($2^{-k}$)** | $1.58\text{b}$ | $1.6\text{-bit}$ | Melhor | Sim | **PoT ($2^{-k}$)** | PWL | Menor | Medido | Mínimo | Menor | Cai em $L$ longo | Cai em $L$ longo |
| **A7: Proposto Completo (`CPoT`)** | $1.58\text{b}$ | $1.6\text{-bit}$ | Melhor | Sim | **CPoT ($1-2^{-k}$)** | **PWL** | Menor | **Mínimo** | **Mínimo** | **Mínimo** | **Alto** | **Alta em $L$ longo** |

### 4.3 Verificação Bit-Exata, Execução na Placa (HPS+FPGA) e Potência Real (Pontos 8, 9, 10 e 11)
1. **Golden Model Bit-Exato (`sw/golden_model_bitexact.py`):** Simulador escrito inteiramente em aritmética inteira de ponto fixo (NumPy `int8`/`int16`/`int32` com *bit-shifts* e *clipping* idênticos ao Verilog). O testbench (`tb_bitmamba_top.v`) compara automaticamente a saída RTL ciclo a ciclo contra o Golden Model exigindo **erro exatamente igual a zero (`0 LSB mismatch`)**.
2. **Potência Estimada com VCD Real + Medição Física:**
   - No Quartus PowerPlay, geraremos arquivos `.vcd` (*Value Change Dump*) a partir da simulação RTL real com entradas do dataset para obter taxas de chaveamento (*toggle rates*) reais em cada nó (em vez do chute padrão de 12.5% *vectorless*).
   - Na execução em placa (Cyclone V SoC), mediremos a potência da placa em *idle* vs. *full load* (via sensor de corrente/medidor em série na alimentação DC) e explicitaremos no texto a diferença metodológica entre potência total da placa vs. telemetria de GPU (`nvidia-smi`).
3. **Integração HPS ARM + FPGA via AXI:** Para validação fim-a-ponta em hardware real, o processador **ARM Cortex-A9 (HPS)** da Cyclone V SoC enviará os tokens de entrada e lerá os estados/saídas do acelerador através da ponte **Lightweight HPS-to-FPGA / FPGA-to-HPS AXI Bridge**, medindo o tempo real com contadores de ciclo de hardware (`PMCCNTR` do ARM + contador de ciclos interno no Verilog).

---

## 5. Estrutura de Diretórios e Arquivos do Projeto (`article/`)

```text
article/
├── PLANEJAMENTO_ESTUDO.md              # Este plano diretor revisado
├── sw/                                 # Pipeline Python: QAT, Golden Model Bit-Exato e Benchmarks
│   ├── models/
│   │   ├── mamba2_ssd_qat.py           # Mamba-2 com H-BitLinear (FWHT no QAT), STE e CPoT Scan
│   │   └── golden_model_bitexact.py    # Modelo inteiro bit-exato idêntico ao hardware RTL
│   ├── train_ablation.py               # Script de treino com 3 seeds para gerar a Tabela de Ablação (A0..A7)
│   ├── long_range_experiment.py        # Experimento de acurácia/MSE vs. comprimento L (PoT vs CPoT)
│   ├── export_quartus_mif.py           # Empacotador Radix-3 (5 trits/byte) -> .mif e vetores .hex para o testbench
│   └── roofline_and_plots.py           # Geração dos gráficos (Roofline Dual-Regime, Pareto DSE, L-scaling)
├── hw/                                 # Código RTL Sintetizável, Testbench e Scripts Quartus
│   ├── rtl/
│   │   ├── radix3_decoder.v            # LUT decoder 8-bit -> 5x trits {-1, 0, +1} em 1 ciclo
│   │   ├── fwht_16_stream.v            # Borboleta Fast Walsh-Hadamard 16-pt sem multiplicadores
│   │   ├── bitlinear_pe.v              # Comparativo: Adder-Tree vs. Table-Lookup (TLMM) vs. Zero-Skip
│   │   ├── baseline_int8_pe.v          # Baseline justo W8A8 usando blocos DSP da Cyclone V
│   │   ├── cpot_selective_scan.v       # Scan Mamba-2 com CPoT {2^-k, 1-2^-k}, PWL SiLU e DSP B*x/C*h
│   │   └── bitmamba_axi_top.v          # Top-level com registradores mapeados em memória (Avalon/AXI-Lite)
│   ├── tb/
│   │   └── tb_bitmamba_bitexact.v      # Testbench de verificação automática contra golden_model_bitexact.py
│   └── quartus/
│       ├── run_dse_synthesis.tcl       # Script TCL que sintetiza todas as variantes (P=8,16,32,64 + INT8 baseline)
│       └── run_vcd_power.tcl           # Script para análise de potência com arquivo .vcd real
└── paper/                              # Artigo de 12 Páginas no Formato SBC (SSCAD / WPerformance)
    ├── sbc-template.sty
    ├── main.tex
    ├── references.bib
    └── figures/
```

---

## 6. Cronograma de Execução Passo a Passo

1. **Etapa 1 — Implementação do Modelo QAT (`H-BitLinear` + `CPoT`) e Golden Model Bit-Exato (`article/sw/`):**
   - Implementar e validar matematicamente a camada `H-BitLinear` (com FWHT no loop de QAT), a discretização `CPoT` ($\{2^{-k}, 1-2^{-k}\}$) e o `golden_model_bitexact.py`.
   - Rodar o experimento de retenção de memória em função de $L$ provando a superioridade do `CPoT` sobre o `PoT` padrão.
2. **Etapa 2 — Implementação RTL Completa + Baseline INT8 no Mesmo Chip + Verificação Bit-Exata (`article/hw/`):**
   - Implementar os módulos Verilog, incluindo o baseline `baseline_int8_pe.v` e as variantes ternárias (`Adder-Tree`, `TLMM`, `Zero-Skip`), e verificar `0 LSB error` no simulador RTL contra o Python.
3. **Etapa 3 — Síntese Automatizada no Intel Quartus (DSE) e Geração dos Dados Reais:**
   - Rodar os scripts do Quartus para extrair ALMs, M10K, DSPs, $F_{\max}$ e potência anotada com `.vcd` para todas as linhas da Tabela de Ablação e da Fronteira de Pareto.
4. **Etapa 4 — Geração dos Gráficos Científicos e Redação Completa do Manuscrito SBC (12 Páginas):**
   - Gerar todas as figuras em formato vetorial (PDF) e redigir o artigo completo em `article/paper/main.tex`.
