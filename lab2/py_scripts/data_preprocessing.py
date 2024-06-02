import pandas as pd # для работы с табличными данными
import numpy as np # для операций линейной алгебры и прочего
from sklearn.preprocessing import StandardScaler # для стандартизации
from sklearn.preprocessing import PowerTransformer  # для степенного преобразования
from sklearn.base import BaseEstimator, TransformerMixin # для создания собственных преобразователей/трансформеров данных
from sklearn.preprocessing import OneHotEncoder# Импортируем One-Hot Encoding от scikit-learn
from sklearn.preprocessing import OrdinalEncoder # для порядкового кодирования
from sklearn.pipeline import Pipeline # для создания pipeline
from sklearn.compose import ColumnTransformer # для преобразования колонок
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# Создаём директории для сохранения данных
os.makedirs('./lab2/train', exist_ok=True)
os.makedirs('./lab2/test', exist_ok=True)

# Загружаем данные в датафрэйм
df = pd.read_csv('lab2/data/train.csv', delimiter = ',')

# Убираем столбцы с пропущенными значениями
df = df.dropna(axis='columns')

# Убираем дублирующий столбец индексов
df.drop(['Id'], axis= 1 , inplace= True )

# Определяем категориальные и числоввые признаки

category_columns = []
num_columns = []

# Делим признаки на группы для дальнейшей обработки

standart_columns = ['OverallQual', 'OverallCond', 'FullBath', 'BedroomAbvGr',
                    'TotRmsAbvGrd', 'GarageCars',  'MoSold', 'YrSold']
power_columns = ['MSSubClass', 'LotArea', 'YearBuilt', 'YearRemodAdd',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF',
                 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',
                 'KitchenAbvGr', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF',
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                 'MiscVal']
rare_standart_columns = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea']
ordinal_columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']
one_hot_columns = ['Neighborhood', 'Exterior1st', 'Exterior2nd']

# определим класс QuantileReplacer, который наследуется от классов BaseEstimator и TransformerMixin из scikit-learn
class QuantileReplacer(BaseEstimator, TransformerMixin):
    # Класс принимает параметр threshold,
    # который определяет относительное пороговое значение
    # для идентификации редких числовых значений
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.quantiles = {}

    def fit(self, X, y=None):
        # вычисляет нижний и верхний квантили для каждого числового признака
        # во входном фрейме данных pandas и сохраняет их в словаре.
        for col in X.select_dtypes(include='number'):
            low_quantile = X[col].quantile(self.threshold)
            high_quantile = X[col].quantile(1 - self.threshold)
            self.quantiles[col] = (low_quantile, high_quantile)
        return self

    def transform(self, X):
        # заменяет редкие числовые значения значениями,
        # основанными на квантилях, хранящихся в словаре
        X_copy = X.copy()
        for col in X.select_dtypes(include='number'):
            low_quantile, high_quantile = self.quantiles[col]
            rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
            if rare_mask.any():
                rare_values = X_copy.loc[rare_mask, col]
                replace_value = np.mean([low_quantile, high_quantile])
                if rare_values.mean() > replace_value:
                    # Если редкое значение выше,
                    # чем среднее значение нижнего и верхнего квантилей,
                    # оно заменяется значением высокого квантиля
                    X_copy.loc[rare_mask, col] = high_quantile
                else:
                    # В противном случае оно заменяется значением нижнего квантиля
                    X_copy.loc[rare_mask, col] = low_quantile
        return X_copy
    
class RareGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05, other_value='Other'):
        self.threshold = threshold
        self.other_value = other_value
        self.freq_dict = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object']):
            freq = X[col].value_counts(normalize=True)
            self.freq_dict[col] = freq[freq >= self.threshold].index.tolist()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in X.select_dtypes(include=['object']):
            X_copy[col] = X_copy[col].apply(lambda x: x if x in self.freq_dict[col] else self.other_value)
        return X_copy

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.target_mean = {}

    def fit(self, X, y):
        if self.cols is None:
            self.cols = X.columns
        for col in self.cols:
            self.target_mean[col] = {}
            X_copy = X.copy()
            X_copy[y.name]=y
            self.target_mean[col] = X_copy.groupby(col)[y.name].mean().to_dict()
        return self

    def transform(self, X):
        for col in self.cols:
            X[col] = X[col].map(self.target_mean[col])
            X[col] = X[col].fillna(np.mean(X[col]))
        return X

# стандартизация

standart_pipeline = Pipeline([('scaler', StandardScaler())])
standart_list = standart_columns

# замена редких событий и стандартизация

rare_standart_pipeline = Pipeline([
    ('QuantReplace', QuantileReplacer(threshold=0.01, )),
    ('scaler', StandardScaler())
])
rare_standart_list = rare_standart_columns

# Степенное преобразование

power_pipeline = Pipeline([('power', PowerTransformer())])
power_list = power_columns

# Порядковое кодирование

ordinal_pipeline = Pipeline([
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('scaler', StandardScaler())
])
ordinal_list = ordinal_columns

# One-hot кодирование

one_hot_pipeline = Pipeline([
    ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
])

one_hot_list = one_hot_columns

# pipeline с числовыми признаками

preprocessors_num = ColumnTransformer(transformers=[
    ('standart_list', standart_pipeline, standart_list),
    ('rare_standart_list', rare_standart_pipeline, rare_standart_list),
    ('power_list', power_pipeline, power_list),
    ])
preprocessors_num

# pipeline со всеми признаками
preprocessors_all = ColumnTransformer(transformers=[
    ('standart_list', standart_pipeline, standart_list),
    ('rare_standart_list', rare_standart_pipeline, rare_standart_list),
    ('power_list', power_pipeline, power_list),
    ('ordinal_list', ordinal_pipeline, ordinal_list),
    ('one_hot_list', one_hot_pipeline, one_hot_list)
])
preprocessors_all

# Разбиваем на тренировочную и валидационную
# не забываем удалить целевую переменную цену из признаков
X_forest, y_forest = df.drop(columns = ['SalePrice']), df['SalePrice']

# разбиваем на тренировочную и валидационную
X_train, X_val, y_train, y_val = train_test_split(X_forest, y_forest,
                                                    test_size=0.3,
                                                    random_state=42)

# Сначала обучаем на тренировочных данных
X_train_prep = preprocessors_all.fit_transform(X_train)
# потом на валидационной
X_val_prep = preprocessors_all.transform(X_val)

one_hot__names = preprocessors_all.transformers_[4][1]['encoder'].get_feature_names_out(one_hot_list)
# объединяем названия колонок в один список (важен порядок как в ColumnTransformer)
columns = np.hstack([standart_list,
                    rare_standart_list,
                    power_list,
                    ordinal_list,
                    one_hot__names])

df_x_train_prep = pd.DataFrame(X_train_prep)
df_x_val_prep = pd.DataFrame(X_val_prep)
df_x_train_prep.columns = columns
df_x_val_prep.columns = columns

# Сохранение обработанных данных
df_x_train_prep.to_csv(f'./lab2/train/x_train.csv', index=False)
df_x_val_prep.to_csv(f'./lab2/test/x_val.csv', index=False)
y_train.to_csv('./lab2/train/y_train.csv', index=False)
y_val.to_csv('./lab2/test/y_test.csv', index=False)