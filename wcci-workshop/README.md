# SUNT Examples

Exemplos de visualização e previsão de demanda no transporte público usando o dataset **SUNT** (Salvador Urban Network of Transportation).

O dataset integra dados de AFC (bilhetagem), AVL (GPS dos veículos), LTI (informações de viagem) e GTFS dos ônibus de Salvador/BA.

**Paper do dataset:** Ferreira et al., *"SUNT: A multimodal urban public transportation dataset"*, Scientific Data, Nature (2025).  
**DOI:** https://www.nature.com/articles/s41597-025-05674-6

---

## Estrutura dos scripts

Cada script é autocontido e grava seus resultados em `outputs/<tema>/`.

| Script | Modelo / Tema |
|---|---|
| `01_viz_timeseries.py` | Visualização de séries temporais (perfis diários, heatmap, sazonalidade, média móvel) |
| `02_viz_graphs.py` | Visualização do grafo da rede (top-20 paradas, centralidade, mapa interativo Folium) |
| `03_forecast_sarima.py` | SARIMA univariado com seleção automática de ordem via `pmdarima.auto_arima` |
| `04_forecast_lstm.py` | LSTM empilhado multivariado (todas as N paradas simultaneamente) |
| `05_forecast_gru.py` | GRU — mesma arquitetura do LSTM com células GRU |
| `06_forecast_transformer.py` | Transformer encoder-only com positional encoding |
| `07_forecast_chronos.py` | Chronos (Amazon) — modelo fundação zero-shot, sem treinamento |
| `08_forecast_gcn.py` | Pure GCN espacial — linha de base para os modelos de grafo |
| `09_forecast_gnn.py` | T-GCN — GCN espacial + GRU temporal (Zhao et al., 2020) |
| `10_forecast_gat.py` | T-GAT — GAT espacial + GRU temporal (Veličković et al., 2018) |

---

## Instalação

```bash
pip install -r requirements.txt
```

**Notas:**

- `torch-geometric` requer instalação separada compatível com a versão do PyTorch instalada — veja [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Os scripts 08, 09 e 10 fazem fallback para uma implementação customizada caso PyG não esteja disponível.
- `chronos-forecasting` requer Python 3.12+.

---

## Como executar

```bash
python 01_viz_timeseries.py
python 03_forecast_sarima.py
python 10_forecast_gat.py
# etc.
```

Os scripts baixam e cacheiam automaticamente os dados em `~/.cache/sunt_dataset/`. Na primeira execução espere o download dos parquets do HuggingFace.

---

## Dados (`utils/data_loader.py`)

O módulo `utils` centraliza o acesso ao dataset via o pacote `suntdataset` (PyPI) ou diretamente pelo HuggingFace (`source="hgface"`).

| Função | Retorno |
|---|---|
| `load_timeseries(freq, top_n, ...)` | Matriz `(T × N)` de embarques por parada — entrada principal dos modelos deep learning |
| `load_boarding(...)` | Eventos de embarque brutos |
| `load_alighting(...)` | Eventos de desembarque estimados |
| `load_od(...)` | Tabela Origem-Destino (loading por parada/viagem) |
| `load_gtfs(table)` | Tabelas estáticas GTFS: `stops`, `trips`, `routes`, `shapes`, `agency` |
| `prepare_sequences(ts, input_len, horizon)` | Janela deslizante → `{X_train, y_val, X_test, scaler, …}` |
| `build_graph_from_od(od)` | `nx.DiGraph` de fluxos entre paradas consecutivas |

**Tipos de dia disponíveis:** `"all"`, `"workdays"`, `"saturdays"`, `"sundays"`

---

## Modelos de grafo (08–10)

Todos os três constroem o grafo da rede a partir dos dados OD:
- **Nós** = paradas de ônibus (top N por volume de embarques)
- **Arestas** = paradas consecutivas na mesma viagem, ponderadas pelo total de passageiros

| Modelo | Componente espacial | Componente temporal |
|---|---|---|
| GCN (08) | GCNConv — pesos fixos pela adjacência normalizada | — (INPUT_LEN como features diretas) |
| T-GCN (09) | GCNConv — pesos fixos | GRU |
| T-GAT (10) | GATConv — **atenção aprendida** por aresta | GRU |

O T-GAT também gera `gat_attention.png` (implementação customizada): heatmap N×N mostrando quais pares de paradas o modelo aprendeu a priorizar.

---

## Hiperparâmetros comuns (deep learning)

| Parâmetro | Valor padrão |
|---|---|
| `INPUT_LEN` | 72 (3 dias horários) |
| `HORIZON` | 24 (previsão 1 dia à frente) |
| `FREQ` | `"1h"` |
| `EPOCHS` | 50–60 |
| `DEVICE` | CUDA se disponível, senão CPU |
| `SEED` | 42 |

Métricas reportadas: **MAE**, **RMSE**, **MAPE** sobre valores na escala original (passageiros/hora).

---

## Outputs gerados

```
outputs/
  viz_timeseries/     perfis diários, heatmap, sazonalidade, média móvel
  viz_graphs/         grafo top-20, centralidade, mapa interativo (.html)
  sarima/             ACF/PACF, forecast com intervalo de confiança
  lstm/               curvas de treino, forecast, lstm_best.pt
  gru/                curvas de treino, forecast, gru_best.pt
  transformer/        forecast, transformer_best.pt
  chronos/            forecast zero-shot
  gcn/                forecast, MAE por nó, gcn_best.pt
  gnn/                forecast, MAE por nó, tgcn_best.pt
  gat/                forecast, MAE por nó, heatmap de atenção, gat_best.pt
```
