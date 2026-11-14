# Revisão de novidade: decaimento sem multiplicador e precisão de estado no *scan* do Mamba

Data: 2026-11-09. Revisor: 1 (com assistência de IA). Status: revisão rápida (*rapid review*), não revisão sistemática completa.

## 1. Método

Protocolo baseado em **Kitchenham & Charters (2007)**, *Guidelines for performing Systematic Literature Reviews in Software Engineering* (padrão em computação), com **snowballing** de **Wohlin (2014)** e relato no formato de fluxo **PRISMA 2020**. Por ter sido feita com busca web (sem exportação de bases como IEEE Xplore, ACM DL ou Scopus), com um único revisor e em um dia, ela se classifica como *rapid review* (Tricco et al., 2015). As limitações estão na Seção 6.

### 1.1 Questão de revisão
Algum trabalho anterior:
- **QR1:** aproxima o coeficiente de decaimento Ā de SSMs/RNNs lineares por potências de dois ou deslocamentos, em especial pelo complemento (Ā ≈ 1 − 2^−k)?
- **QR2:** analisa a precisão do estado inteiro (zona morta, bits de guarda, arredondamento) em *scan* de SSM, e seu efeito na acurácia?
- **QR3:** elimina o multiplicador do decaimento numa unidade de *scan* em hardware (RTL/FPGA/ASIC) e mede o custo?
- **QR4:** treina (QAT) o modelo com o *scan* inteiro bit-exato do hardware?

### 1.2 Fontes e strings de busca
Fonte: busca web geral (indexa arXiv, IEEE, ACM, CVF, ACL, OpenReview, MDPI, Semantic Scholar), 14 buscas, modos padrão e estendido.

| Faceta | String (resumida) |
|---|---|
| Decaimento PoT | `Mamba state space model quantization decay coefficient power-of-two shift hardware` |
| Aceleradores | `Mamba FPGA accelerator multiplier-free selective scan exponential approximation` ; `Mamba accelerator ASIC scan unit exp discretization hardware 2026` |
| Precisão do estado | `state space model fixed-point hidden state precision integer recurrence ... Mamba` ; `integer-only Mamba inference quantized selective scan hidden state INT8 INT16` |
| RNNs lineares | `linear RNN RWKV RetNet decay quantization FPGA ASIC shift-based decay` ; `S4 S5 LRU fixed-point FPGA neuromorphic SSM quantization` |
| Forma exata | `"1 - 2^{-k}" OR "complementary power-of-two" quantization decay recurrent` ; `Mamba QAT hardware integer scan forget gate power-of-two "1-2^" decay` |
| PTQ de Mamba | `Quamba2 Mamba-2 quantization A dt state decay INT8 scan kernel` |
| DSP clássico | `exponential moving average bit shift fixed-point leaky integrator deadband rounding` |
| Snowballing | `QMamba LtSQ cited ...` ; `ViM-Q ... state transition shift` ; `eMamba ... exp approximation hidden state bit width` |

### 1.3 Critérios
- **Inclusão:** trata da quantização ou implementação em hardware do *scan*/recorrência de SSM ou RNN linear (Mamba, S4/S5, RWKV, RetNet, GRU/LSTM em PoT), ou da implementação em ponto fixo de filtros recursivos de 1ª ordem por deslocamento.
- **Exclusão:** trata só das projeções lineares; não descreve como o decaimento é calculado; páginas secundárias (blogs de resumo, agregadores) quando a fonte primária estava disponível.

## 2. Fluxo (PRISMA 2020, contagens aproximadas)

```
Registros identificados (14 buscas × ~9 resultados)      ≈ 125
  − duplicatas e páginas secundárias (agregadores/resumos) ≈ 70
Registros triados por título/resumo                        ≈ 55
  − fora do escopo (só projeções, outro domínio)           ≈ 30
Trabalhos avaliados em mais detalhe                          25
  ├─ texto completo lido: QMamba, MARCA-v2, SSDi8 (parcial), HFRWKV (parcial)
  └─ resumo/trechos: demais
Incluídos na síntese                                         22 (Tabela 3)
```

## 3. Extração de dados

| # | Trabalho | Plataforma | Como faz Ā·h | Precisão do estado / zona morta | QAT? | QR |
|---|---|---|---|---|---|---|
| 1 | **QMamba** (Li et al., arXiv 2501.13624, 2025) | GPU (PTQ simulada), visão | **Â = 1 − 2^−q, q = round(−log2(1−A)); Â·h = h − (h >> q)** (LtSQ), escolhido por tensor se mediana(A) > α=0,9 | Estado em 4–8 bits com escala por grupo temporal (TGQ); **não** discute zona morta, guarda nem arredondamento do deslocamento | Não (PTQ) | **QR1 sim** |
| 2 | LightMamba (Wei et al., DATE 2025) | FPGA (VCK190/U280) | Multiplicação; PoT só nos fatores de re-escala | — | Não | — |
| 3 | FastMamba (2505.18975) | FPGA (VC709) | exp por 2^v + PWL de 8 segmentos; multiplicação | — | Não | — |
| 4 | MARCA (ISCA 2024) | ASIC simulado | exp "fast biased" (shift + add) e multiplicação | — | Não | — |
| 5 | MARCA-v2 (IEEE TCAD 2025) | ASIC 28 nm | idem; "complementar" = esparsidade do SSM, não decaimento | — | Não | — |
| 6 | SSDi8 (2608.21952, 2026) | GPU (A5000/Orin) | Decaimento em ponto fixo dividido por constante 2^7 (deslocamento só nesta divisão) | Estado INT8 com escala por (H,P), acumulação INT32, arredondamento; sem zona morta | Não (PTQ) | parcial |
| 7 | eMamba (ACM TECS 2025) | FPGA ZCU102 + ASIC GF 22 nm | Aproximação da exp; escalas PoT | Larguras por análise da aplicação; sem zona morta | Treino com aproximações | parcial QR3 |
| 8 | Mamba-X (2508.02977) | ASIC (visão) | *Scan* sistólico; exp em SFU | — | Não | — |
| 9 | ViM-Q (FCCM 2026) | FPGA ZCU102 | Multiplicador Ā; SSM mantido em alta precisão | — | Não | — |
| 10 | LowRank-SSM (2608.02954) | FPGA | Foco nas projeções | — | — | — |
| 11 | HFRWKV (2601.02135) | FPGA U50/U280 | exp por 2^Y com LUT de 256 entradas | ~9 bits; não discute | Não | — |
| 12 | Quamba / Quamba2 (2024/ICML 2025) | GPU | INT8 PTQ; não detalha Ā | — | Não | — |
| 13 | PTQ4VM (WACV 2025) | GPU | — | Estado INT8 derruba Vim-Ti para 6,8% | Não | QR2 parcial |
| 14 | Q-Mamba (ACL Findings 2025) | GPU | — | Quantiza o *cache* de estado | Não | QR2 parcial |
| 15 | Nemotron 3 Super (2604.12374) | GPU | — | *Cache* SSM INT16 só funciona com escalas por bloco | Não | QR2 parcial |
| 16 | Ternary Mamba (2606.18114) | GPU | FP16 | Erro acumula como Σ Ā^(T−t) ε_t; motiva QAT | Sim (pesos) | — |
| 17 | Q-S5 / QS4D / IMSSA / SSMs pequenos p/ borda (2024–2026) | Loihi 2, IMC | Ā quantizado uniformemente | A e estado são os mais sensíveis | Sim (QAT) | QR2 parcial |
| 18 | 2QGRU (Electronics 2026) | FPGA | PoT nos pesos de GRU | — | — | — |
| 19 | Filtros EMA/integrador com vazamento por deslocamento (literatura de DSP: Lyons/DSPRelated, dspguru, notas de implementação) | — | **y += (x − y) >> k**, i.e. coeficiente 1 − 2^−k | **Zona morta, solução com bits extras e arredondamento (1 << (k−1)) são conhecidas** | — | **QR1, QR2 (fora de SSM)** |

## 4. Síntese

**QR1: não é novidade.** A forma Ā ≈ 1 − 2^−k com h − (h >> k) já foi proposta para Mamba pelo **QMamba (LtSQ)**, em janeiro de 2025. Ela também é o filtro EMA clássico por deslocamento em DSP. O CPoT acrescenta só dois detalhes: o ramo bimodal por token, que escolhe entre 2^−k e 1 − 2^−k pelo limiar 0,625 (o QMamba escolhe por tensor entre LtSQ e uniforme), e o código (m,k) obtido por tabela indexada por j (Δ = 2^−j). São refinamentos, não uma contribuição principal.

**QR2: lacuna confirmada no contexto de SSM.** Nenhum trabalho de SSM encontrado analisa a zona morta do decaimento por deslocamento, os bits de guarda ou o arredondamento, nem mede o efeito disso na acurácia de um modelo treinado. Isso inclui o QMamba, que usa justamente o deslocamento com estado de 4–8 bits. Os trabalhos de SSM só relatam que "o estado é sensível". O fenômeno em si é conhecido em DSP, então a contribuição é **trazer e quantificar** esse efeito no *scan* seletivo, não descobri-lo.

**QR3: lacuna provável.** Nenhum acelerador encontrado elimina o multiplicador de Ā·h. Todos calculam a exponencial (LUT, PWL ou shift) e depois multiplicam. Nenhum mede o custo de uma unidade com decaimento por deslocamento contra uma com multiplicador.

**QR4: lacuna provável.** Os trabalhos que usam deslocamento no decaimento (QMamba) são PTQ com *fake quantization*. Não encontramos QAT com o *scan* inteiro bit-exato de um RTL verificado.

## 5. Veredito e reposicionamento recomendado

O artigo **não pode** apresentar o CPoT como proposta própria. Contribuições defensáveis:

1. **Análise de precisão do estado no *scan* inteiro (QR2):** a zona morta do decaimento 1 − 2^−k congela canais de memória longa, o efeito aparece em modelos treinados, e 32 bits com 10 bits de guarda e arredondamento resolvem. É a contribuição mais forte, e a que o QMamba não cobre.
2. **Unidade de *scan* sem multiplicador no decaimento, medida em hardware (QR3):** RTL verificado bit a bit; Quartus com 48 contra 80 DSP; ASIC Nangate45 em andamento.
3. **QAT com *scan* inteiro bit-exato e comparação com PoT simples numa tarefa que exige esquecimento gradual (QR4):** evidência em acurácia de que o complemento é necessário, que o QMamba não apresenta.
4. Refinamentos: código bimodal por token e tabela (m,k) indexada por j.

Mudanças no texto:
- Creditar a forma 1 − 2^−k ao QMamba [Li et al., 2025] e ao EMA por deslocamento em DSP. Chamar o esquema de "complemento em potência de dois (como no LtSQ do QMamba)", com o CPoT como a variante bimodal por token.
- Incluir QMamba, SSDi8, eMamba, ViM-Q, Mamba-X, MARCA-v2, HFRWKV, PTQ4VM e Q-S5/QS4D nos trabalhos relacionados e na Tabela 1.
- Na Fundamentação, citar o conhecimento clássico de zona morta e arredondamento em filtros IIR de ponto fixo como base da análise.
- Mudar o título e o resumo de "propomos o CPoT" para "o que é preciso para um *scan* inteiro sem multiplicador funcionar".

## 6. Ameaças à validade

- **Busca:** busca web, não exportação de bases; resultados dependem do ranqueamento do buscador. Faltam IEEE Xplore e ACM DL com strings booleanas completas e Google Scholar com *backward snowballing* completo das referências do QMamba e do LightMamba.
- **Um revisor**, sem dupla triagem nem medida de concordância.
- **Leitura:** só 4 trabalhos lidos em texto completo; os demais por resumo e trechos. Afirmações negativas ("não discute zona morta") valem para o que foi lido.
- **Recência:** trabalhos de 2026 podem não estar indexados.
- **Próximo passo** para fechar a revisão antes da submissão: (i) buscas nas bases IEEE/ACM/Scopus com as mesmas strings; (ii) *forward snowballing* das citações do QMamba no Google Scholar; (iii) leitura integral de eMamba, ViM-Q e SSDi8.

## 7. Fase 2 (2026-11-09): busca em base bibliográfica e *snowballing* para frente

### 7.1 Por que OpenAlex
IEEE Xplore e Scopus exigem chave de API ou login institucional. Usamos o **OpenAlex** (api.openalex.org), que indexa os metadados de IEEE, ACM, Elsevier, Springer e arXiv e aceita strings booleanas sobre título e resumo. Para o *snowballing* para frente usamos o **Semantic Scholar** (citações do QMamba e do LightMamba). Período: a partir de 2023-12 (lançamento do Mamba); S6–S8 a partir de 2018.

| ID | String (título + resumo) | Registros |
|---|---|---|
| S1 | (mamba OR "state space model" OR "selective scan") AND (accelerator OR FPGA OR ASIC) | 96 |
| S2 | (mamba OR "state space model" OR "selective scan") AND quantization AND ("power-of-two" OR shift OR "bit-shift") | 10 |
| S3 | (mamba OR "state space model" OR "selective scan") AND ("fixed-point" OR integer) AND (state OR recurrence) | 83 |
| S4 | (mamba OR "state space model") AND ("hidden state" OR "state precision") AND (quantization OR "bit-width") | 6 |
| S5 | ("linear recurrence" OR "linear attention" OR RWKV OR "gated recurrent") AND (FPGA OR accelerator) AND (decay OR "power-of-two" OR shift) | 3 |
| S6 | (deadband OR "dead band" OR "dead zone" OR "limit cycle") AND (recurrent neural OR SSM OR mamba OR RNN) AND (quantization OR "fixed-point") | 8 |
| S7 | ("power-of-two" OR shift) AND (decay OR "forget gate" OR leak) AND (spiking OR recurrent OR "state space") AND (hardware OR FPGA OR ASIC) | 46 |
| S8 | (mamba OR "state space model") AND ("multiplier-free" OR "multiplication-free" OR "shift-add") | 2 |
| F1 | Citações do QMamba (Semantic Scholar) | 5 |
| F2 | Citações do LightMamba (Semantic Scholar) | 33 |

### 7.2 Fluxo
```
Registros identificados (S1–S8 + F1–F2)                    292
  − duplicatas (entre strings e com a fase 1)               ≈ 60
Registros triados por título                               ≈ 230
  − fora do escopo (controle, Kalman, finanças, biologia,
    visão sem hardware, Zenodo sem revisão)                 ≈ 190
Resumos lidos                                                 38
  − só projeções/memória/dataflow, sem detalhar Ā·h           24
Novos incluídos                                               14
```
A string S3 é pouco específica ("state space model" + "fixed-point" traz controle e filtragem); S6 e S7 não retornaram nenhum trabalho sobre zona morta em SSM/RNN quantizada.

### 7.3 Novos trabalhos incluídos

| Trabalho | Plataforma | Decaimento Ā·h | Estado / precisão | Ameaça à novidade |
|---|---|---|---|---|
| **Fuketa & Hioki, "Hardware-Friendly Fully Quantized Mamba-2 Model and Its FPGA-Based Accelerator"** (IEEE Access 2026, 10.1109/ACCESS.2026.3696226) | FPGA Ultra96-V2 | Resumo: exp substituída por "power-of-two functions"; unidade vetor-vetor em DSP (INT8×INT8) | Pesos ternários, ativações INT8; QAT em 2 estágios | **A mais próxima. Texto completo não acessível (IEEE bloqueia acesso automatizado; sem preprint).** O resumo sugere que Ā é uma potência de dois e que o produto Ā·h ainda usa DSP; não menciona complemento, zona morta nem precisão do estado. **Conferir o PDF manualmente.** |
| Hfqmamba2: Zhou et al., "An Efficient FPGA-Based Hardware Accelerator of Fully Quantized Mamba-2" (FCCM 2025) | FPGA | Não detalhado no resumo (LUT segmentada para RMSNorm) | Quantização híbrida completa | Baixa (mesmo grupo do HFRWKV, que usa LUT para exp) |
| LUT-SSM (Yoo et al., ISSCC 2026) | ASIC 28 nm FD-SOI | Operações elemento a elemento fundidas; FP | — | Nenhuma |
| SSMA (Dong et al., ISCAS 2025) | ASIC | Unidades não lineares de baixa complexidade | — | Nenhuma |
| STEAM-SSM (ISCAS 2026) | FPGA U250 | FP16 | FP16 | Nenhuma |
| MambaOPU (DAC 2025) | FPGA *overlay* | Multiplicações elemento a elemento | — | Nenhuma |
| SpecMamba, ToMamba, ViMA, S²Mamba, MedMambaLite-v2, HotBa, EMPSA, aceleradores de vídeo Mamba (2025–2026) | FPGA/ASIC | Foco em esparsidade, *dataflow*, decodificação especulativa | — | Nenhuma |
| KSPiM (65 nm), VGA (MICRO 2024), SSM-RDU, Fine-Grained Fusion, XAMBA | ASIC/NPU | Convolução global, fusão, NPU | — | Nenhuma |
| Mamba para MIMO (Lund, 2026) | RTL | Fluxo ponto fixo Python → C++ bit-exato → RTL | Ponto fixo | Baixa (metodologia semelhante, sem análise de decaimento) |
| PACE (Haller et al., arXiv 2610.29545) | GPU | — | Estado em 4 bits pós-treino; arredondamento por token acumula erro | QR2 parcial: armazenamento do estado, não zona morta do decaimento |
| ReplaySSM (Liou et al., com T. Dao, arXiv 2610.13778) | GPU | — | Separa precisão de leitura e de recorrência; erro de arredondamento do estado realimenta a recorrência | QR2 parcial: confirma que a precisão da recorrência importa |
| Dual-scale SSM (tese KTH, 2026) | NoC | — | INT8 recorta *outliers* e o erro se acumula no estado | QR2 parcial |
| PTQ para ViM com *K-scaled* (MLSP 2025) | GPU | Reparametrização do estado | — | Baixa |
| Quantizing Small-Scale SSMs for Edge AI (ICONS 2025) | — | A e estado mais sensíveis; QAT recupera | — | Já na fase 1 |

### 7.4 Conclusão da fase 2
- **O veredito da fase 1 se mantém.** QR1 (forma 1 − 2^−k) continua creditada ao QMamba e ao DSP. QR2 (zona morta, bits de guarda e arredondamento no *scan* seletivo, e o efeito na acurácia) segue sem trabalho correspondente. PACE, ReplaySSM e a tese da KTH reforçam que a precisão do estado importa, mas tratam do armazenamento e da quantização do estado, não do decaimento por deslocamento.
- **Novo trabalho próximo:** Fuketa & Hioki (IEEE Access 2026) também treina com QAT um Mamba-2 totalmente inteiro, troca a exponencial por potência de dois e implementa em FPGA. Precisa ser citado. Pelo resumo, o Ā·h continua numa unidade DSP e não há forma de complemento nem análise do estado; **isso precisa ser confirmado no PDF**.
- Nenhum trabalho mede uma lane sem multiplicador no decaimento contra uma com multiplicador, nem em FPGA nem em ASIC.

## Fontes
- QMamba: https://arxiv.org/abs/2501.13624
- LightMamba: https://arxiv.org/abs/2502.15260
- FastMamba: https://arxiv.org/html/2505.18975v3
- MARCA: https://arxiv.org/pdf/2409.11440
- MARCA-v2: https://dai.sjtu.edu.cn/my_file/pdf/0a61722b-778e-459c-a5fe-5b01ea5e3661.pdf
- SSDi8: https://arxiv.org/html/2608.21952
- eMamba: https://arxiv.org/abs/2508.10370v1 , https://dl.acm.org/doi/10.1145/3762190
- Mamba-X: https://arxiv.org/html/2508.02977v1
- ViM-Q: https://arxiv.org/html/2605.01935v1
- LowRank-SSM: https://arxiv.org/pdf/2608.02954
- HFRWKV: https://arxiv.org/abs/2601.02135
- Quamba: https://arxiv.org/abs/2410.13229 ; Quamba2: https://proceedings.mlr.press/v267/chiang25a.html
- PTQ4VM: https://openaccess.thecvf.com/content/WACV2025/papers/Cho_PTQ4VM_Post-Training_Quantization_for_Visual_Mamba_WACV_2025_paper.pdf
- Q-Mamba: https://aclanthology.org/2025.findings-acl.551/
- Nemotron 3 Super: https://arxiv.org/pdf/2604.12374
- Ternary Mamba: https://arxiv.org/abs/2606.18114v1
- Q-S5: https://www.researchgate.net/publication/381471438_Q-S5_Towards_Quantized_State_Space_Models
- QS4D: https://arxiv.org/pdf/2507.06079v1
- SSMs pequenos para borda: https://arxiv.org/pdf/2506.12480
- IMSSA: https://arxiv.org/pdf/2412.20215
- 2QGRU: https://www.mdpi.com/2079-9292/15/4/722
- DSP (EMA por deslocamento, zona morta, arredondamento): https://www.dsprelated.com/showarticle/72.php , https://dspguru.com/dsp/tricks/fixed-point-dc-blocking-filter-with-noise-shaping , https://tttapa.github.io/Pages/Mathematics/Systems-and-Control-Theory/Digital-filters/Exponential%20Moving%20Average/C++Implementation.html
