import numpy as np
import os
import pandas as pd
from numpy import random



# Создаём директории для сохранения данных
os.makedirs('./train', exist_ok=True)
os.makedirs('./test', exist_ok=True)

def gen_data(n_samples=5, # всего признаков + целевая переменная
             n_features=5, # всего значений признака
             p_anomaly=0.1, # средний процент аномалий
             scale_size=5):    # среднее значение отклонения
    
    data = np.empty((n_samples,n_features), int) #пустой массив для данных
    # массив с процентома аномалий в каждом признаке
    p_anomaly_array = np.random.normal(loc=p_anomaly, 
                                       scale=0.05, 
                                       size=n_samples-1)
    loc_array =  np.random.randint(10, 100, size=(n_samples-1))
    
    for i in range (n_samples-1):
        n_anomaly = int(n_features*p_anomaly_array[i])
        if (n_anomaly < 0): n_anomaly = -1*n_anomaly
        data_clean = np.random.normal(loc=loc_array[i], scale=scale_size, size=(n_features-n_anomaly))
        rand = random.randint(11, 1000)
        anomaly = np.random.normal(loc=rand,scale=(0.3*rand), size=n_anomaly)
        data[i] = np.concatenate((data_clean, anomaly), axis=0)
        #random.shuffle(data[i])
    #data = np.round(data, 2)
    data[n_samples-1] = np.random.normal(loc=18, scale=scale_size, size=n_features)
    return data.reshape(n_features,n_samples)

def gen_df(n_samples_all=5, # всего признаков + целевая переменная
           n_features_all=1000, # всего значений
           p_train = 0.3): # процент на тренировочную выборку
    n_train_all = int(n_features_all*p_train)
    df_train = pd.DataFrame(gen_data(n_samples_all, n_features=(n_features_all-n_train_all)))
    df_test = pd.DataFrame(gen_data(n_samples_all, n_train_all))
    columns_name = []
    for i in range(n_samples_all-1):
        name = "Features"+str(i)
        columns_name.append(name)
    columns_name.append('label')
    df_train.columns = columns_name
    df_test.columns = columns_name
    return df_train, df_test 

df_train, df_test = gen_df(5, 1000, 0.3)

# Сораняем тренировочный набор
df_train.to_csv(f'./train/train.csv', index=False)

# Сохраняем тестовый набор
df_test.to_csv(f'./test/test.csv', index=False)