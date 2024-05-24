import joblib #для сохранения модели
import os #для работы с ОС
import pandas as pd #для работы с таблицами
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error as mse #
from sklearn.metrics import r2_score #

def calculate_metric(model_pipe, X, y, metric=r2_score):
    y_model = model_pipe.predict(X)
    return metric(y, y_model)

# считываем данные для тренировки и тестирования
X_train = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                   "data/train/X_train.csv"
                                   ))
y_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                   "data/train/y_train.csv"
                                   )).squeeze()
X_test = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  'data/test/X_val.csv'
                                  ))
y_test = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  'data/test/y_val.csv'
                                  ))

# обучаем модель на тренировочных данных
model = LogisticRegression(random_state=42)
model.fit(X_train.values, y_train.values)

# расчитываем и выводим mse и r2 на тренировочных данных
print(f"r2 на тренировочной выборке: {calculate_metric(model, X_train.values, y_train.values):.4f}")

print(f"mse на тренировочной выборке: {calculate_metric(model, X_train.values, y_train.values, mse):.4f}")

# сохраняем модель
if not os.path.isdir(os.path.join(os.path.dirname(__file__),'model')):
        os.mkdir(os.path.join(os.path.dirname(__file__),'model'))
joblib.dump(model, os.path.join(os.path.dirname(__file__),
                                "model/LogisticRegression.pkl"))

# расчитываем и выводим mse и r2 на тестовых данных
print(f"r2 на валидационной выборке: {calculate_metric(model, X_test.values, y_test.values):.4f}")
print(f"mse на валидационной выборке: {calculate_metric(model, X_test.values, y_test.values, mse):.4f}")