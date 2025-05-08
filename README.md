# SUNT - Salvador Urban Network Transportation

## 🗺️ Overview

First and foremost, thanks for your interest in using our dataset, referred to as Salvador Urban Network Transportation (SUNT), and benchmarks.

Here you will find all codes, models, and data used in the manuscript “Salvador Urban Network Transportation (SUNT): A Spatiotemporal dataset on the public transportation”.

More details about our data, codes, and models are shown in each folder.

The following figure illustrates our dataset is geographically distributed in Salvador (Brazil).

<center><img src="images/graphs-SSA.png" width=500px/></center>

On the behalf of all of the authors, we appreciate your interest in our data, code, and models, and hope they are useful to your research.

---

## 🗂️ Repository structure

The organization of this repository is:

> - **data** : raw and graph-based data
> - **data_design** : contains source codes developed to create our datasets and train learning models 
> - **docs**: database documentation
> - **images**: plots with statistics of the dataset attributes
> - **integration**: example of integrating other databases with the dataset
> - **models** : contains frozen models used to predict different tasks using SUNT
> - **outputs**/: weigths and results
> - **stats**: notebook with data statistics.
---


## :floppy_disk: Dataset

### The full dataset is stored on the [Mendelay Data](https://data.mendeley.com/datasets/85fdtx3kr5/1).

 - **RAW**
   - AVL (Automatic Vehicle Location)
   - Automatic Fare Collection (AFC)
   - General Transit Feed Specification (GTFS)
   - Local Trip Information (LTI)
- **Built** 
  - Boarding [Hugging Face](https://huggingface.co/datasets/labiaufba/PublicTransportationSunt)
  - Alighting [Hugging Face](https://huggingface.co/datasets/labiaufba/PublicTransportationSunt)
  - Origin-Destination (OD) [Hugging Face](https://huggingface.co/datasets/labiaufba/PublicTransportationSunt)



More dataset info, see  [dataset doc](docs/datasets.md). 

##  📝 Citation

If you find the dataset useful for your research, please consider citing:



```latex
@dataset{SUNT2025,
  author       = {Marcos Vinícius dos Santos Ferreira and Matheus Carvalho de Souza and Tatiane Nogueira Rios and Islame Felipe da Costa Fernandes and Danilo Oliveira Andrade and Joao Gama and Albert Bifet and Ricardo Rios},
  title        = {Salvador Urban Network Transportation (SUNT)},
  year         = {2025},
  publisher    = {Mendeley Data},
  version      = {1},
  doi          = {10.17632/85fdtx3kr5.1},
  url          = {https://data.mendeley.com/datasets/85fdtx3kr5/1}
}
```

## 📃 License

This project is licensed under the CC BY 4.0 License.
