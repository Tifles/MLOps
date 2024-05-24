import pandas as pd
from sklearn.preprocessing import StandardScaler



df_train = pd.read_csv("./train/train.csv")
df_test = pd.read_csv("./test/test.csv")
# Разделение данных на признаки и целевую переменную
X_train, y_train = df_train.drop(columns=['label']), df_train['label']
X_test, y_test = df_test.drop(columns=['label']), df_test['label']
    
# Инициализация и обучение стандартизатора
scaler = StandardScaler()
scaler.fit(X_train)

# Выполним преобразование
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
df_x_train_scaled = pd.DataFrame(X_train_scaled)
df_x_test_scaled = pd.DataFrame(X_test_scaled)

# Сохранение обработанных данных
df_x_train_scaled.to_csv(f'./train/x_train_standart.csv', index=False)
df_x_test_scaled.to_csv(f'./test/x_test_standart.csv', index=False)
y_train.to_csv('./train/y_train.csv', index=False)
y_test.to_csv('./test/y_test.csv', index=False)