import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# Загрузка тестовых данных
X_test = pd.read_csv('./test/x_test_standart.csv')
y_test = pd.read_csv('./test/y_test.csv')

# Загрузка обученной модели
model = joblib.load('./models/linear_regression_model.pkl')

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Вычисление метрик
mse = mean_squared_error(y_test.values.ravel(), y_pred)
print(f'MSE после теста: {mse}')