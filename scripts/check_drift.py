import pandas as pd
import os
import sys

try:
    # --- ИМПОРТЫ ДЛЯ EVIDENTLY 0.6.0 ---
    # В версии 0.6 Report находится в evidently.report
    from evidently.report import Report
    # Пресеты находятся в evidently.metric_preset
    from evidently.metric_preset import DataDriftPreset
    
    print("Evidently 0.6.0 imports successful")
except ImportError as e:
    print("Ошибка импорта Evidently:", e)
    sys.exit(1)


def run_drift_report():
    print("Начало анализа дрифта данных...")

    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    # Создание синтетических данных (если их нет)
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Данные не найдены, создаю синтетические данные...")
        os.makedirs("data/processed", exist_ok=True)
        
        dummy_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        dummy_data.to_csv(train_path, index=False)

        dummy_data_drifted = dummy_data.copy()
        dummy_data_drifted['feature1'] *= 1.5
        dummy_data_drifted.to_csv(test_path, index=False)

    # Загрузка данных
    reference = pd.read_csv(train_path).drop(columns=['target'], errors='ignore')
    current = pd.read_csv(test_path).drop(columns=['target'], errors='ignore')

    # Генерация отчёта
    print("Генерация отчёта Evidently...")
    
    # В версии 0.6 используется этот синтаксис
    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=None  # В 0.6.0 None работает корректно для автоопределения
    )

    # Сохранение результатов
    os.makedirs("reports", exist_ok=True)
    
    html_path = "reports/data_drift_report.html"
    json_path = "reports/data_drift_report.json"
    
    # 1. Сохранение HTML (метод save_html работает стабильно)
    report.save_html(html_path)
    
    # 2. Сохранение JSON
    # В старых версиях метода save_json() могло не быть, 
    # поэтому надежнее использовать report.json() и сохранить файл вручную
    with open(json_path, 'w') as f:
        f.write(report.json())

    print(f"Отчёты созданы:\n - {html_path}\n - {json_path}")


if __name__ == "__main__":
    run_drift_report()
