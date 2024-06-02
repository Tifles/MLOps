import joblib #для сохранения модели
import streamlit as st
import os #для работы с ОС

#готовим функцию загрузки моделей
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(os.path.dirname(__file__), "model/LogisticRegression.pkl"))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), "model/StandartScaler.pkl"))
    return model, scaler

# Загружаем предварительно обученных моделей
model, scaler = load_model()

#Заголок
st.title('Определение сорта Ириса')

sepal_length = st.slider("Какова длина чашелистика?", 4.0, 8.0, 6.2, 0.1)
sepal_width = st.slider("Какова ширина чашелистика?", 1.0, 5.0, 3.4, 0.1)
petal_length = st.slider("Какова длина лепестка?", 1.0, 7.0, 5.4, 0.1)
petal_width = st.slider("Какова ширина лепестка?", 0.1, 3.0, 2.3, 0.1)
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
input_data_scaled = scaler.transform(input_data)
prediction_index = model.predict(input_data_scaled)[0]
class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
prediction_index_int = int(prediction_index)
result = class_names[prediction_index_int]
st.success(result)