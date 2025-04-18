# Learners

This fold presents examples of models that can be used to explore our dataset. Such models are organized into the following folds:

> - **edge-classification** - Models designed to classify edges in graphs.
> 	- `DataDesignerEdgeClassification.ipynb`: Example of loading, pre-processing and dividing the data set with cross validation.
> 	- `Example-Edge-Classification-BCELoss.ipynb`: Example of training and validating models with cross validation.
> - **node-classification** - Models designed to classify nodes in graphs.
> 	- `DataDesingNodeClassification.ipynb`: Example of loading, pre-processing and dividing the data set with cross validation.
> 	- `Example-Node-Classification.ipynb`: Example of training and validating models with cross validation.
> - **node-regression** - Models designed to predict node values in graphs.
> 	- `load_data.ipynb`: Example of loading, pre-processing and dividing the data set into train and test.
> 	- `Exemple.ipynb`: Example of training, forecasting and validating models
> 	- `sarima_grid.ipynb`:  SARIMA grid search.
> 	- `Results-Node-Regression.ipynb`: Compile results of cross validation and computer validation metrics.



### All examples have common Python modules:

- `data.py` : Module to load and pre-process data.
- `main.py`: Performs the model training and validation pipeline.
- `models.py`: It has the architecture of the models.
-  `val.py`: Has validation functions.
- `workflow.py`: Present the functions for training and testing models.
