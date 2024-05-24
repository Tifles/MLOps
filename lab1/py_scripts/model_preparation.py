import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import joblib
import os
import sys


# Функция обучения моодели и подсчёта метрики

def traing_and_calculat(df_x, df_y):
    
     # Создание и обучение модели линейной регрессии
    model = LogisticRegression()

    # Обучение
    model.fit(df_x, df_y.values.ravel())

    # Предсказание на тренировочной выборке
    y_pred = model.predict(df_x)

    # Считаем метрики
    mse = mean_squared_error(df_y, y_pred)

    # Выводим полученные метрики
    
    print(f'MSE: {mse}')
    
    

    return model


# Создаём директорию модели
os.makedirs('./models', exist_ok=True)

# Считываем тренировочные данные
df_x = pd.read_csv("./train/x_train_standart.csv")
df_y = pd.read_csv("./train/y_train.csv")

# Тренируемся
model = traing_and_calculat(df_x, df_y)

#Сохраняем обученную модель
joblib.dump(model, './models/linear_regression_model.pkl')