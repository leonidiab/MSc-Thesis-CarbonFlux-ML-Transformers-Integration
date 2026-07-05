# Classificação de NEE entre Sítios (Cross-Site) - Tese de Mestrado _Classificação de fluxos de carbono integrando transformers e métodos clássicos de aprendizado de máquina_

Classificação do fluxo de carbono (NEE) utilizando modelos baseados em árvore e transformers com teste cruzado entre sítios (torres de fluxo) australianos.

## Início Rápido

### Instalação

```bash
pip install -e ".[dev]"
```

### Treinamento

Treine um modelo com um sítio de holdout específico:

```bash
# Modelos baseados em árvore (XGBoost, LightGBM, RandomForest, etc.)
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model trees

# FT-Transformer
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model ft_transformer

# TabTransformer
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model tabtransformer
```

### Teste Cruzado (Cross-Site Testing)

Teste um modelo treinado no sítio de holdout correspondente:

```bash
python scripts/test_cross_site.py \
    --model-dir outputs/holdout_wombat/trees/f1_macro/with_BOM/run_01 \
    --test-site wombat
```
### Painel de Teste Interativo (Dashboard)

Você pode iniciar o painel interativo web para testar visualmente os melhores modelos salvos com novos conjuntos de dados:

```bash
python scripts/run_app.py
```

Isso iniciará automaticamente um servidor Flask local seguro em `http://127.0.0.1:5000` e abrirá a interface no seu navegador padrão. Recursos principais:
* **Hub de Modelos**: Varredura dinâmica que exibe informações e métricas de todos os modelos salvos em `outputs/`.
* **Predição em Lote (CSV)**: Arraste e solte arquivos CSV, visualize as previsões na tabela e baixe o resultado final rotulado. Caso o dataset de teste contenha a coluna real `target_class`, a interface calcula métricas de acurácia e plota a matriz de confusão correspondente.
* **Previsão em Tempo Real**: Teste dados individuais através de formulários dinâmicos com sliders baseados nos limites reais de cada variável do modelo.

Para mais informações sobre a arquitetura da interface, API REST e salvaguardas de segurança, consulte o [Manual do Usuário Completo](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/documentation/USER_GUIDE_PT_BR.md#4-painel-de-testes-interativo-dashboard-web).

### Opções da CLI

```bash
python scripts/train.py --help
python scripts/test_cross_site.py --help
```

## Reproduzindo Experimentos

Execute todos os 15 experimentos (3 tipos de modelo × 5 sítios de holdout):

```bash
for site in cumberland robson_creek_queensland tumbarumba whroo wombat; do
    for model in trees ft_transformer tabtransformer; do
        python scripts/train.py \
            --config configs/cross_site/holdout_${site}.yaml \
            --model ${model}
    done
done
```

## Estrutura do Projeto

```
├── configs/                    # Configurações de experimentos em YAML
│   ├── base.yaml               # Padrões compartilhados
│   ├── cross_site/             # Configurações por sítio de holdout
│   └── models/                 # Configurações por tipo de modelo
├── data/processed/             # Datasets pré-processados dos sítios
│   ├── cumberland/
│   ├── robson_creek_queensland/
│   ├── tumbarumba/
│   ├── whroo/
│   └── wombat/
├── src/nee_classification/     # Pacote Python principal
│   ├── config/                 # Schemas de configuração Pydantic v2
│   ├── data/                   # Carregamento e pré-processamento de dados
│   ├── models/                 # Classes de treinamento (árvores, FT-T, TabT)
│   ├── evaluation/             # Métricas e geração de relatórios
│   ├── tuning/                 # Seleção de features
│   ├── runner/                 # Worker e orquestrador
│   ├── artifacts/              # Persistência de modelos
│   └── utils/                  # Sementes (seeds), detecção de hardware do sistema
├── scripts/                    # Pontos de entrada da CLI
│   ├── train.py
│   └── test_cross_site.py
├── tests/                      # Suite de testes com pytest
├── outputs/                    # Resultados do treinamento (ignorado no git)
└── pyproject.toml              # Configuração de build PEP 517/518
```

## Dados

Cinco sítios de torres de fluxo (flux towers) australianas com 28 variáveis (features) cada:

| Sítio | Linhas | Descrição |
|------|------|-------------|
| Cumberland | 138 | Planície de Cumberland (Cumberland Plain) |
| Robson Creek Queensland | 46 | Floresta tropical úmida (Tropical rainforest) |
| Tumbarumba | 642 | Floresta esclerófila úmida (Wet sclerophyll forest) |
| Whroo | 184 | Savana/floresta seca (Dry woodland) |
| Wombat | 230 | Floresta temperada (Temperate forest) |

**Variáveis (Features):** Sensoriamento remoto do MODIS (Fpar, LAI, GPP, ET, LE, LST, refletância de superfície), dados meteorológicos do BOM (Bureau Of Meteorology) (temperatura, precipitação) e variáveis temporais (ano, mês, semana do ano, dia do ano).

**Alvo (Target):** Classificação binária (`NS` = Não sumidouro, `S` = Sumidouro de carbono).

## Desenho Experimental

**Teste cruzado entre sítios (Cross-site testing):** Para cada experimento, 4 sítios são utilizados para treinamento e validação e o sítio restante é deixado de fora (holdout) para teste independente. Isso avalia a capacidade de generalização do modelo para localizações geográficas não vistas durante o treino.

**Otimização:** Cada experimento executa N rodadas independentes (Optuna trials) através de M métricas, com e sem as features do BOM, gerando uma comparação abrangente.

## Dados

Este projeto utiliza dados do **[FLUXNET2015 Dataset](https://fluxnet.org/)**.

**Licença de uso:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
**Citação:** Pastorello, G., Trotta, C., Canfora, E. *et al.* The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data. *Sci Data* **7**, 225 (2020). [https://doi.org/10.1038/s41597-020-0534-3](https://doi.org/10.1038/s41597-020-0534-3)

**Fontes adicionais:** 
* **MODIS:** Courtesy of the NASA EOSDIS Land Processes Distributed Active Archive Center (LP DAAC).
* **[Bureau of Meteorology](https://www.bom.gov.au/) (BoM):** © Commonwealth of Australia, Bureau of Meteorology. Licensed under CC BY 3.0 AU.

## Licença

MIT