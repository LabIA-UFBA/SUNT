# Getting Started

## 1. Access the Data

| Data type | Source |
|---|---|
| Raw data (AVL, AFC, GTFS, LTI) | [Mendeley Data](https://data.mendeley.com/datasets/85fdtx3kr5/1) |
| Processed data (Boarding, Alighting, OD) | [Hugging Face](https://huggingface.co/datasets/labiaufba/PublicTransportationSunt) |
| Code & models | [GitHub](https://github.com/LabIA-UFBA/SUNT) |

---

## 2. Clone the Repository

```bash
git clone https://github.com/LabIA-UFBA/SUNT.git
cd SUNT
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Load Processed Data (Hugging Face)

```python
from datasets import load_dataset

boarding  = load_dataset("labiaufba/PublicTransportationSunt", "boarding")
alighting = load_dataset("labiaufba/PublicTransportationSunt", "alighting")
od        = load_dataset("labiaufba/PublicTransportationSunt", "od")
```

---

## 5. Load Data Locally (Pandas)

```python
import pandas as pd

avl_lines    = pd.read_csv("data/raw/avl_lines.csv")
avl_vehicles = pd.read_csv("data/raw/avl_vehicles.csv")
afc          = pd.read_csv("data/raw/afc.csv")
lti          = pd.read_csv("data/raw/lti.csv")

boarding  = pd.read_csv("data/processed/boarding.csv")
alighting = pd.read_csv("data/processed/alighting.csv")
od        = pd.read_csv("data/processed/od.csv")
```

---

## 6. Load the Graph

```python
import pandas as pd
import networkx as nx

nodes = pd.read_csv("data/graph/nodes.csv")
edges = pd.read_csv("data/graph/edges.csv")

G = nx.from_pandas_edgelist(
    edges,
    source="src",
    target="dst",
    edge_attr=["distance", "average_speed", "trip_time", "loading"],
    create_using=nx.DiGraph()
)
```

---

## 7. Explore the Sample Notebook

```bash
jupyter notebook docs/dataloader_sample.ipynb
```

---

## Next Steps

- Understand each table in the [Dataset Overview](dataset/overview.md)
- Consult the full [Data Dictionary](data-dictionary.md)
- Follow the [Tutorials](tutorials/loading-data.md) for specific use cases
