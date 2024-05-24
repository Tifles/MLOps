import joblib #для сохранения модели
import os #для работы с ОС
import pandas as pd #для работы с таблицами
from sklearn.datasets import load_iris # для загрузки датасета Ирис
from sklearn.model_selection import train_test_split # для разбиения на 
# тренировочную и тестовую выборки  
from sklearn.preprocessing import StandardScaler # для стандартизации данных

# Скачиваем dataset
dataset_iris = load_iris()

# Преобразуем в датафрейм
df_iris=pd.DataFrame(dataset_iris.data, columns=dataset_iris.feature_names)
df_iris['target'] = dataset_iris.target

# Разделяем на тренировочную и тестовую выборки 
# и сохраняем целевые значения в файлы
X, y = df_iris.drop(columns='target'), df_iris['target']
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42
                                                    )
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

# Выполняем стандартизацию и сохраняем в файлы
scaler = StandardScaler()
scaler.fit(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled)
X_test_scaled_df = pd.DataFrame(X_test_scaled)

if not os.path.isdir(os.path.join(os.path.dirname(__file__),'data')):
        os.mkdir(os.path.join(os.path.dirname(__file__),'data'))

if not os.path.isdir(os.path.join(os.path.dirname(__file__),'data/train')):
        os.mkdir(os.path.join(os.path.dirname(__file__),'data/train'))

if not os.path.isdir(os.path.join(os.path.dirname(__file__),'data/test')):
        os.mkdir(os.path.join(os.path.dirname(__file__),'data/test'))

X_train_scaled_df.to_csv(os.path.join(os.path.dirname(__file__),
                                   'data/train/X_train.csv'
                                   ),
                      index=False
                      )

y_train_df.to_csv(os.path.join(os.path.dirname(__file__),
                            'data/train/y_train.csv'
                            ),
               index=False
               )

X_test_scaled_df.to_csv(os.path.join(os.path.dirname(__file__),
                                  'data/test/X_val.csv'
                                  ),
                     index=False
                     )
y_test_df.to_csv(os.path.join(os.path.dirname(__file__),
                           'data/test/y_val.csv'
                           ),
              index=False
              )
if not os.path.isdir(os.path.join(os.path.dirname(__file__),'model')):
        os.mkdir(os.path.join(os.path.dirname(__file__),'model'))
joblib.dump(scaler, os.path.join(os.path.dirname(__file__),
                                "model/StandartScaler.pkl"))