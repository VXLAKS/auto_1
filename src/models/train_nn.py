import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. Определение архитектуры (добавлен Dropout для 'промышленного' вида)
class CreditNet(nn.Module):
    def __init__(self, input_size):
        super(CreditNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_and_export():
    # Создаем папку для моделей, если ее нет
    os.makedirs("models", exist_ok=True)

    # 2. Загрузка и подготовка данных
    print("⏳ Загрузка данных...")
    train_df = pd.read_csv("data/processed/train.csv")
    X_raw = train_df.drop("target", axis=1).values.astype('float32')
    y = train_df["target"].values.reshape(-1, 1).astype('float32')

    # Важно: нейросети требуют масштабирования признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler сохранен в models/scaler.pkl")

    # 3. Инициализация модели и обучения
    model = CreditNet(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # 4. Цикл обучения
    print("🚀 Начало обучения...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(torch.from_numpy(X))
        loss = criterion(outputs, torch.from_numpy(y))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # 5. Сохранение весов PyTorch
    torch.save(model.state_dict(), "models/model_nn.pth")
    
    # 6. Экспорт в ONNX
    print("📦 Экспорт в ONNX...")
    model.eval() # Переключаем в режим инференса (важно для Dropout и BatchNorm)
    
    input_size = X.shape[1]
    dummy_input = torch.randn(1, input_size)
    
    # Используем стабильные параметры для последующего квантования
    torch.onnx.export(
        model, 
        dummy_input, 
        "models/model.onnx", 
        export_params=True,
        opset_version=14, # 14 - самая стабильная версия для квантования
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        # Убираем динамические оси для решения ошибки ShapeInferenceError при квантовании
    )
    
    print("✅ Модель обучена и сохранена: models/model.onnx")
    print("⚠️ Примечание: Экспорт выполнен со статичным размером (batch=1) для стабильности.")

if __name__ == "__main__":
    train_and_export()