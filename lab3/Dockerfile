# Base image
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер в /usr/src/app
COPY src /app
COPY requirements.txt /app

# Устанавливаем библиотеки
RUN pip install -r requirements.txt

# Обучаем и запускаем модель
CMD python src/data_preprocessing.py && python src/model_train.py  && streamlit run src/main.py