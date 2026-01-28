import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def process_data():
    # Загружаем сырые данные (предполагаем, что ты положил свой csv в data/raw/)
    try:
        # Если исходный файл называется по-другому, поменяй имя здесь
        df = pd.read_csv('data/raw/UCI_Credit_Card.csv')
    except FileNotFoundError:
        print("Ошибка: Файл data/raw/UCI_Credit_Card.csv не найден.")
        sys.exit(1)

    # Если вдруг колонка называется по-старому, переименуем. Если уже target - ок.
    if 'default.payment.next.month' in df.columns:
        df.rename(columns={'default.payment.next.month': 'target'}, inplace=True)

    # Удаляем ID, он не несет полезной инфы для модели
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    # Разбиваем на train и test
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

    # Сохраняем
    train.to_csv('data/processed/train.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    print("Данные успешно обработаны и сохранены в data/processed/")

if __name__ == "__main__":
    process_data()