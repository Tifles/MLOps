# EN
[![RU]](/HW1/README.ru.md)

# Laboratory work №1 

Creating an automatic machine learning project pipeline.

## Project content:
1. py_scripts is a directory with python script files.

1.1. data_creation.py - creates various datasets describing a certain process. There are several sets. Some data contains anomalies or noises. Some of the sets are saved in the "train" folder, the other part is in the "test" folder.
1.2. model_preprocessing.py - performs data preprocessing using sklearn.preprocessing.Standard Scale r.
1.3. model_preparation.py - which creates and trains a machine learning model based on the built data from the "train" folder.
1.4. model_testing.py - checks the machine learning model on the constructed data from the "test" folder.

2. pipeline.sh - bash script that runs all python scripts sequentially.
