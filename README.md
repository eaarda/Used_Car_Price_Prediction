# Used Car Price Prediction
This repository contains used car price prediction using Artificial Neural Network in Keras Sequential model.


### Introduction
I used "Car Features and MSRP" dataset from kaggle.  This dataset initially had 11914 rows. The dataset included Make, Model, Year, Engine Fuel Type, Engine HP, Engine Cylinders, Transmission Type, Driven Wheels, Number of Doors, Market Category, Vehicle Size, Vehicle Style, highway MPG, city mpg, Popularity, MSRP(price).

The dataset avaible at:
https://www.kaggle.com/CooperUnion/cardataset

- Rapor.pdf -> report
-  ann.ipynb -> all codes in Jupyter notebook
-  activation.py -> only includes activation functions
- neuralNetwork.py -> not completed yet and not required
-  requirements.txt -> list of required libraries and versions

#### Prerequisites
- Python 3.8.6
- Jupyter Notebook

#### Model Result
Keras Sequential Model (lr=0.001, epochs=100, batch_size=128)
- Test Accuracy:  95.42 %
- Macro Precision: 95.06 %
- Macro Recall: 92.07 %
- Macro F1-score: 93.45 %
- Auc Score: 92.07 %
