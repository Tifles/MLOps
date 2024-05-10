import joblib
import numpy as np # библиотека Numpy для операций линейной алгебры и прочего
import numpy as np # библиотека Numpy для операций линейной алгебры и прочего
import os
import pandas as pd # Библиотека Pandas для работы с табличными данными
from sklearn.ensemble import RandomForestRegressor # Случайный Лес для Регрессии от scikit-learn
from sklearn.model_selection import ShuffleSplit # при кросс-валидации случайно перемешиваем данные
from sklearn.model_selection import cross_validate # функция кросс-валидации от Scikit-learn

from sklearn.metrics import mean_squared_error as mse # метрика MSE от Scikit-learn
from sklearn.metrics import r2_score

# Создаём директорию модели
os.makedirs('./HW2/models', exist_ok=True)

x_train_prep=pd.read_csv('./HW2/train/x_train.csv')
y_train=pd.read_csv('./HW2/train/y_train.csv')

#Создаём объект класса
model_rf_forest = RandomForestRegressor(n_estimators=150,
                                max_depth=10,
                            oob_score=True
                              )

#обучаем
model_rf_forest.fit(x_train_prep, y_train.values.ravel())

#Сохраняем обученную модель
joblib.dump(model_rf_forest, './HW2/models/rf_forest_model.pkl')
