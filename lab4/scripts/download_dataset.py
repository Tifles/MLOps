from catboost.datasets import titanic
import pandas as pd
import os

#   Проверяем наличие каталога для датасета. если нет, то создаём его
if not os.path.isdir(os.path.join(os.path.dirname(__file__), "datasets")):
        os.mkdir(os.path.join(os.path.dirname(__file__), "datasets"))

# Получаем датасет
titanic_train, titanic_test = titanic()
titanic_df = pd.concat([titanic_train, titanic_test], ignore_index=True)

# Создание отдельной папки для данных
os.makedirs('datasets', exist_ok=True)

# Сохраняем данные
titanic_df.to_csv(os.path.join(os.path.dirname(__file__), 'datasets/titanic_df.csv')