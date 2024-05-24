import joblib
import os
import pandas as pd # Библиотека Pandas для работы с табличными данными
from sklearn.ensemble import RandomForestRegressor # Случайный Лес для Регрессии от scikit-learn

from sklearn.metrics import mean_squared_error as mse # метрика MSE от Scikit-learn

# Создаём директорию модели
os.makedirs('./lab2/models', exist_ok=True)

x_train_prep=pd.read_csv('./lab2/train/x_train.csv')
y_train=pd.read_csv('./lab2/train/y_train.csv')

#Создаём объект класса
model_rf_forest = RandomForestRegressor(n_estimators=150,
                                max_depth=10,
                            oob_score=True
                              )

#обучаем
model_rf_forest.fit(x_train_prep, y_train.values.ravel())

#Сохраняем обученную модель
joblib.dump(model_rf_forest, './lab2/models/rf_forest_model.pkl')
