import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift_report():
    print(" Анализ дрифта данных...")
    
    # 1. Загружаем эталонные данные (на которых учились) и текущие
    # Для имитации возьмем train и test
    if not os.path.exists("data/processed/train.csv") or not os.path.exists("data/processed/test.csv"):
        print("Файлы данных не найдены!")
        return

    reference = pd.read_csv("data/processed/train.csv").drop(columns=['target'])
    current = pd.read_csv("data/processed/test.csv").drop(columns=['target'])

    # 2. Создаем отчет
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    # 3. Сохраняем
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/data_drift_report.html")
    print("Отчет создан: reports/data_drift_report.html")

if __name__ == "__main__":
    run_drift_report()