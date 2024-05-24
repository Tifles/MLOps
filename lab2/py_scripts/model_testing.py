import joblib
import pandas as pd # Библиотека Pandas для работы с табличными данными
import numpy as np # библиотека Numpy для операций линейной алгебры и прочего
import numpy as np # библиотека Numpy для операций линейной алгебры и прочего


from sklearn.model_selection import ShuffleSplit # при кросс-валидации случайно перемешиваем данные
from sklearn.model_selection import cross_validate # функция кросс-валидации от Scikit-learn

from sklearn.metrics import mean_squared_error as mse # метрика MSE от Scikit-learn
from sklearn.metrics import r2_score

def calculate_metric(model_pipe, X, y, metric = r2_score):
    """Расчет метрики.
    Параметры:
    ===========
    model_pipe: модель или pipeline
    X: признаки
    y: истинные значения
    metric: метрика (r2 - по умолчанию)
    """
    y_model = model_pipe.predict(X)
    return metric(y, y_model)

def cross_validation (X, y, model, scoring, cv_rule):
    """Расчет метрик на кросс-валидации.
    Параметры:
    ===========
    model: модель или pipeline
    X: признаки
    y: истинные значения
    scoring: словарь метрик
    cv_rule: правило кросс-валидации
    """
    scores = cross_validate(model,X, y,
                      scoring=scoring, cv=cv_rule )
    print('Ошибка на кросс-валидации')
    DF_score = pd.DataFrame(scores)
    #print('\n')
    print(DF_score.mean()[2:])


# Загрузка тестовых данных
x_train_prep=pd.read_csv('./lab2/train/x_train.csv')
x_val_prep=pd.read_csv('./lab2/test/x_val.csv')
y_train=pd.read_csv('./lab2/train/y_train.csv')
y_val=pd.read_csv('./lab2/test/y_test.csv')

# Загрузка обученной модели
model = joblib.load('./lab2/models/rf_forest_model.pkl')

print(f"r2 на тренировочной выборке: {calculate_metric(model, x_train_prep, y_train):.4f}")
print(f"r2 на валидационной выборке: {calculate_metric(model, x_val_prep, y_val):.4f}")

print(f"mse на тренировочной выборке: {calculate_metric(model, x_train_prep, y_train, mse):.4f}")
print(f"mse на валидационной выборке: {calculate_metric(model, x_val_prep, y_val, mse):.4f}")

scoring_reg = {'R2': 'r2',
           '-MSE': 'neg_mean_squared_error',
           '-MAE': 'neg_mean_absolute_error',
           '-Max': 'max_error'}

cross_validation (x_train_prep, y_train.values.ravel(),
                  model,
                  scoring_reg,
                  ShuffleSplit(n_splits=5, random_state = 42))