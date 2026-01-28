import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import sys

# Определяем схему валидации
schema = DataFrameSchema({
    "LIMIT_BAL": Column(float, Check.greater_than_or_equal_to(0)),
    "AGE": Column(int, Check.between(18, 100)),
    "SEX": Column(int, Check.isin([1, 2])),
    "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6])),
    "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3])),
    "target": Column(int, Check.isin([0, 1])),
    "PAY_0": Column(int, Check.between(-2, 9)),
})

def validate_data(filepath):
    print(f"Запуск валидации через Pandera для {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        # Проверка данных по схеме
        schema.validate(df, lazy=True)
        print(f"Валидация прошла успешно: {filepath}")
    except pa.errors.SchemaErrors as err:
        print(f"Ошибка валидации в файле {filepath}:")
        # Выводим только конкретные ошибки для краткости
        print(err.failure_cases[['column', 'check', 'failure_case']])
        sys.exit(1)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_data('data/processed/train.csv')
    validate_data('data/processed/test.csv')