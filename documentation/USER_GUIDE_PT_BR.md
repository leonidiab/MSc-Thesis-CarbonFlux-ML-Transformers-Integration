# Projeto Classificação de NEE - Manual do Usuário Completo

Este guia fornece instruções completas sobre como configurar, treinar, testar e avaliar de forma interativa os modelos dentro do framework de classificação de NEE (Net Ecosystem Exchange - Fluxo Líquido do Ecossistema).

---

## 1. Estrutura do Projeto

* **`configs/`**: Arquivos de configuração YAML que definem os experimentos (datasets, sítios de holdout, hiperparâmetros dos modelos).
* **`data/processed/`**: Dados pré-processados organizados por sítio (ex: `data/processed/wombat/full_dataset.csv`).
* **`documentation/`**: Manuais do usuário e relatórios de segurança do projeto.
* **`outputs/`**: Modelos salvos, encoders de pré-processamento e relatórios de treino.
* **`scripts/`**: Scripts de orquestração e execução:
  * `train.py`: Inicia o treinamento e a otimização de hiperparâmetros com o Optuna.
  * `test_cross_site.py`: Teste dos modelos nos conjuntos de dados de holdout via CLI.
  * `run_app.py`: Inicia o painel interativo web para testes e inferência visual.
* **`src/nee_classification/`**: Pacote Python principal:
  * `data/`: Carregadores de dados e pipelines de limpeza.
  * `models/`: Classes de treinamento (árvores e redes neurais).
  * `web/`: Código do servidor Flask, templates HTML e arquivos estáticos (CSS/JS).

---

## 2. Configurações de Experimentos (YAML)

Os experimentos são estruturados em arquivos YAML. Eles determinam as divisões de dados, alocação de hardware e seleção de variáveis.

Exemplo de configuração básica (`configs/base.yaml`):
```yaml
name: base_experiment
random_seed_base: 42
n_runs: 10
n_trials: 600
optimization_metrics: [accuracy, f1_macro]
primary_metric: f1_macro
data:
  base_dir: data/processed
  target_col: target_class
  features_to_drop_always: [NEE_VUT_REF, TIMESTAMP]
```

---

## 3. Treinamento de Modelos

Utilize o script `scripts/train.py` para rodar os experimentos de treino e ajuste de hiperparâmetros. O script salva o melhor modelo geral na pasta `outputs/`.

**Treinar Modelos de Árvores (XGBoost, LightGBM, Random Forest):**
```bash
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model trees
```

**Treinar Modelos de Deep Learning (FT-Transformer ou TabTransformer):**
```bash
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model ft_transformer
python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model tabtransformer
```

*Nota: Para modelos de Deep Learning em computadores com recursos limitados de memória RAM, passe o parâmetro `--max-workers 1` para evitar travamentos de Out-Of-Memory.*

---

## 4. Painel de Testes Interativo (Dashboard Web)

O projeto possui uma interface gráfica interativa baseada em web para exploração de modelos e inferência. Esta ferramenta permite selecionar qualquer modelo treinado que esteja na pasta `outputs/`, realizar previsões de instâncias individuais usando sliders adaptados para cada variável ou carregar arquivos CSV inteiros para obter previsões em lote.

### 4.1 Estrutura de Arquivos

A interface web está estruturada em uma arquitetura clássica Flask (Backend) + HTML/CSS/JS (Frontend):
* **Script de Execução**: [scripts/run_app.py](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/scripts/run_app.py)
* **Backend Flask**: [src/nee_classification/web/server.py](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/server.py)
* **Template HTML**: [src/nee_classification/web/templates/index.html](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/templates/index.html)
* **Lógica JS (Frontend)**: [src/nee_classification/web/static/app.js](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/static/app.js)
* **Estilização CSS (Design Premium)**: [src/nee_classification/web/static/style.css](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/static/style.css)

### 4.2 Requisitos de Instalação

Para executar o painel, garanta que os pacotes necessários estão instalados em seu ambiente virtual:
```bash
pip install flask pandas scikit-learn joblib xgboost matplotlib seaborn
# Opcional (se for carregar modelos FT-Transformer ou TabTransformer)
pip install pytorch-tabular pytorch-lightning
```

### 4.3 Inicialização

Execute o comando a seguir na raiz do projeto:
```bash
python scripts/run_app.py
```
Esse script inicia o servidor Flask em `http://127.0.0.1:5000` e abre o navegador automaticamente na página inicial da aplicação. Se não houver modelos treinados na pasta `outputs/`, o script emitirá um aviso lembrando de treinar pelo menos um modelo com o `scripts/train.py`.

### 4.4 Funcionalidades da Interface

O dashboard é composto por três seções principais organizadas em um layout de tema "Eco-Friendly Premium" (verde-menta/floresta e painéis de vidro translúcidos):

1. **Hub de Modelos (Model Selection Hub)**:
   - Realiza uma busca dinâmica no diretório `outputs/` ao carregar a página.
   - Apresenta os modelos na forma de cartões exibindo: nome simplificado do modelo (`XGBoost`, `FT-Transformer` ou `TabTransformer`), o sítio geográfico de validação/holdout, se utiliza variáveis meteorológicas do BoM, a métrica de otimização principal e o melhor score de validação alcançado.
   - O usuário escolhe o modelo desejado clicando em **"Load Model"**, carregando-o instantaneamente na memória do servidor.

2. **Predição em Lote (Batch Prediction via CSV)**:
   - Suporta upload de arquivos por arrastar e soltar (drag and drop) ou seleção de arquivo local.
   - **Tabela de Prévia**: Apresenta as 10 primeiras linhas dos dados combinadas com as predições de classificação geradas pelo modelo.
   - **Download de Resultados**: Permite exportar a planilha inteira contendo os dados originais adicionados das colunas de predição (`predicted_class` com `"S"` ou `"NS"`) e as probabilidades calculadas (`probability_S` e `probability_NS`).
   - **Métricas e Matriz de Confusão**: Se o CSV carregado contiver a coluna original `target_class` (com valores rotulados reais), o backend calcula automaticamente a Acurácia, F1-Score Macro, Precisão e Recall do lote. Além disso, gera dinamicamente e plota uma imagem base64 de uma Matriz de Confusão customizada integrada ao painel.

3. **Predição Individual (Single Instance Real-time Entry)**:
   - Configura de forma dinâmica um formulário interativo de acordo com as variáveis exigidas pelo modelo carregado.
   - **Campos Numéricos e Sliders**: Mapeiam os limites mínimos e máximos reais encontrados no conjunto de treinamento para impedir que o usuário insira valores fisicamente impossíveis ou fora de escala.
   - **Seletores Categóricos**: Permitem a escolha de fatores estruturados como Ano, Mês, Dia do Ano e Semana do Ano.
   - **Predição Instantânea**: Clicando em **"Predict"**, a predição é enviada via API em tempo real e exibe o badge de classificação final (**Sink (S)** ou **No Sink (NS)**) com uma barra indicativa de probabilidade para cada classe.

---

## 5. Endpoints da API REST (Backend)

O servidor Flask disponibiliza duas rotas principais para comunicação com o cliente:

### 5.1 Listar Modelos Treinados
* **Rota**: `/api/models`
* **Método**: `GET`
* **Descrição**: Varre a pasta `outputs/` por arquivos de metadados `info.json` e retorna uma lista formatada de todos os modelos disponíveis que possuem artefatos de inferência válidos (`model.joblib` ou diretório `model_final`).
* **Exemplo de Resposta**:
  ```json
  [
    {
      "id": "holdout_wombat/trees/f1_macro/with_BOM/run_01",
      "model_type": "trees",
      "metric_optimized": "f1_macro",
      "best_val_score": 0.842,
      "use_bom": true,
      "features": ["Fpar", "LAI", "Ta_F", "Precip_F"],
      "holdout_site": "wombat",
      "class_mapping": {"NS": 0, "S": 1}
    }
  ]
  ```

### 5.2 Executar Predição
* **Rota**: `/api/predict`
* **Método**: `POST`
* **Tipo de Requisição**: `multipart/form-data`
* **Parâmetros**:
  - `model_id` (string, obrigatório): O caminho relativo do modelo (e.g., `holdout_wombat/trees/...`).
  - `input_type` (string, obrigatório): Deve ser `"single"` ou `"file"`.
  - `data` (string JSON, obrigatório para `input_type="single"`): Dicionário contendo os valores de features da instância.
  - `file` (arquivo CSV, obrigatório para `input_type="file"`): Arquivo CSV com as colunas de entrada correspondentes.
* **Respostas**:
  - **Inundação de Instância Única**:
    ```json
    {
      "prediction": "S",
      "probabilities": {
        "NS": 0.1245,
        "S": 0.8755
      }
    }
    ```
  - **Execução em Lote (CSV)**:
    ```json
    {
      "predictions_count": 230,
      "preview": [ ... ],
      "csv_data": "Fpar,LAI,predicted_class,...\n0.6,1.2,S,...\n",
      "has_ground_truth": true,
      "metrics": {
        "accuracy": 0.891,
        "f1_macro": 0.887,
        "precision_macro": 0.89,
        "recall_macro": 0.885,
        "details": { ... }
      },
      "confusion_matrix_b64": "iVBORw0KGgoAAA..."
    }
    ```

---

## 6. Salvaguardas de Robustez e Segurança

 A exposição de modelos de Machine Learning via interface web exige controles de segurança integrados ao código de [src/nee_classification/web/server.py](file:///C:/Users/leoni/Documents/Github/MSc-Thesis-CarbonFlux-ML-Transformers-Integration-DEV/src/nee_classification/web/server.py):

* **Prevenção de Path Traversal**: Caminhos fornecidos pelo cliente para carregar modelos são verificados usando caminhos absolutos resolvidos contra a pasta `outputs/`. Se o caminho tentar escapar desse diretório (utilizando `..`), o backend lança uma exceção `PermissionError` e retorna um status `403 Forbidden`.
* **Desserialização Segura**: O backend Flask nunca aceita o upload de arquivos de modelos (como `.joblib` ou pesos do PyTorch) de usuários via HTTP. Ele carrega exclusivamente modelos armazenados localmente e criados pelo próprio pipeline de treino interno.
* **Tratamento de Colunas Faltantes**: Caso uma entrada (seja formulário manual ou CSV de lote) esteja sem alguma coluna exigida pelo modelo (como variáveis numéricas ou categóricas internas do `pytorch-tabular`), o backend preenche automaticamente esses campos com valores padrão (`0` ou `0.0`) e reordena os dados de maneira transparente para manter a compatibilidade com a matriz de normalização e pesos.
* **Tratamento de Valores Nulos (NaN)**: No upload de arquivos, se houver dados faltantes nas features do modelo, eles são automaticamente imputados com o valor mediano da respectiva coluna do CSV antes da inferência, evitando falhas ou travamentos de processamento do scikit-learn e do PyTorch.
* **Cabeçalhos de Segurança HTTP**: Configurações de Content Security Policy (CSP), X-Frame-Options, X-XSS-Protection e X-Content-Type-Options são injetadas em todas as respostas HTTP do servidor.
