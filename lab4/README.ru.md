# RU
[![EN]](./README.md)

# Лабораторная работа №4

Работа с улитой DVC,  предназначеной для правления данными.

## Описание файлов:
* scripts/download_dataset.py - файл для загрузки датасета;
* scripts/dataset_modif_age.py - файл заполняет пропуски столбца `Age` средними значениями;
* scripts/dataset_add_one_hot_sex.py - файл выролняет one_hot кодирования столбца `Sex`;

## Этапы работы:

1. Подготовка набора данных;
2. Включение отслеживания dvc за набором данных;
3. Изменение набора данных и сохранение результатов с помощью git + dvc;
4. Переключение между версиями наборов данных с помощью git + dvc;

В качестве хранилища используем [Google Drive folder](https://drive.google.com/drive/folders/12KZSI3PtauAQHk53OXmrK7J5JI0nkwpB?usp=sharing)

![alt text](image.png)

Историю изменений можно увидеть через `git log --oneline`:

```shell
5099ecb  modified datasets one_hot_encoding
62453cc  modified Age mean
db71a5c  Modif dataset
7497140  Changes to be committed:       modified:   .dvc/config         new file:   lab4/scripts/dataset_add_one_hot_sex.py     new file:   lab4/scripts/dataset_modif_age.py   modified:   lab4/scripts/download_dataset.py
```
Версии датасета хранятся в коммитах: `7497140`, `db71a5c`, `62453cc` и `5099ecb`

Переключение между коммитами выполняются командой `git checkout <commit-id>`.
Для загрузки датасета необходимо выполнить`dvc pull`.

Для возврата к  последней версии:
```shell
git checkout master
dvc pull
```
