import pandas as pd
import numpy as np
import joblib

def calculate_psi(expected, actual, buckets=10):
    """Рассчитывает Population Stability Index"""
    def scale_by_bin(samples, bins):
        hist, _ = np.histogram(samples, bins=bins)
        return hist / len(samples)

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, buckets + 1)

    e_percents = scale_by_bin(expected, bins)
    a_percents = scale_by_bin(actual, bins)

    # Убираем нули для стабильности логарифма
    e_percents = np.clip(e_percents, 0.0001, 1)
    a_percents = np.clip(a_percents, 0.0001, 1)

    psi_value = np.sum((e_percents - a_percents) * np.log(e_percents / a_percents))
    return psi_value

def run_monitoring():
    model = joblib.load("models/model.pkl")
    train_df = pd.read_csv("data/processed/train.csv").drop("target", axis=1)
    
    # Имитируем новые данные (просто берем тест)
    new_data = pd.read_csv("data/processed/test.csv").drop("target", axis=1)

    # Получаем вероятности дефолта
    train_probs = model.predict_proba(train_df)[:, 1]
    new_probs = model.predict_proba(new_data)[:, 1]

    psi = calculate_psi(train_probs, new_probs)
    print(f"📊 Monitoring Report:")
    print(f"Population Stability Index (PSI): {psi:.4f}")
    
    if psi < 0.1:
        print("No significant drift detected.")
    elif psi < 0.25:
        print("Moderate drift detected. Consider retraining soon.")
    else:
        print("Significant drift! Model performance might be degraded.")

if __name__ == "__main__":
    run_monitoring()