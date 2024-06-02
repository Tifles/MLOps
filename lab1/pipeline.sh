#!/bin/bash

# Функция для создания виртуального окружения

create_venv() {
    local   venv_name=${1:-".venv"}
    python3 -m venv "$venv_name"
    echo "The virtual environment '$venv_name' has been created."  
}

# Функция активации виртуального окружения

activate_venv() {
    local venv_name=${1:-".venv"}
    if [ ! -d "$venv_name" ]; then
        create_venv
    fi
    if [ -z "$VIRTUAL_ENV" ]; then
        source "./$venv_name/bin/activate"
        echo "Virtual environment '$venv_name' is activated."
    else
        echo "The virtual environment has already been activated."
    fi
}

# Функция для установки зависимостей из requirements.txt
install_dependency() {
    if [ ! -f "requirements.txt" ]; then
        echo "File requirements.txt not found."
        return 1
    fi

    # Check if all dependencies from requirements.txt are installed
    for package in $(cat requirements.txt | cut -d '=' -f 1); do
        if ! pip freeze | grep -q "^$package=="; then
            echo "Dependency installation..."
            pip install -r requirements.txt
            echo "Dependencies installed."
            return 0
        fi
    done

    echo "All dependencies are already installed."
}

# Активируем или создаём и активируем виртуальную среду
activate_venv

# Устанавливаем зависимости
install_dependency

#Выполняем генерацию данных
python py_scripts/data_creation.py

#Выполняем преобразование данных
python py_scripts/model_preprocessing.py

# Тренируем модель
python py_scripts/model_preparation.py

# Тестируем модель
python py_scripts/model_testing.py