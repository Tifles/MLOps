import os
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

# Считываем данные
titanic_df = pd.read_csv(os.path.join(os.path.dirname(__file__),'../datasets/titanic_df.csv'))
df = titanic_df

# Инициализация и применение OneHotEncoder
encoder = OneHotEncoder()
one_hot_encoded = encoder.fit_transform(df[['Sex']])

# Создание DataFrame из закодированных данных
one_hot_df = pd.DataFrame(one_hot_encoded.toarray(), columns=encoder.get_feature_names_out(['Sex']))

# Объединение с исходным DataFrame
df_encoded = pd.concat([df, one_hot_df], axis=1)
df_encoded = df_encoded.drop(columns=['Sex'])

df_encoded.to_csv(os.path.join(os.path.dirname(__file__),'../datasets/titanic_df.csv'), index=False)