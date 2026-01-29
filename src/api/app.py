from fastapi import FastAPI
import onnxruntime as ort
import numpy as np
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Credit Scoring Industrial API")

# 1. Загружаем артефакты (теперь их два: скалер и ONNX-модель)
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/model.onnx"

try:
    scaler = joblib.load(SCALER_PATH)
    # Создаем сессию ONNX Runtime для быстрого инференса на CPU
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print("✅ Модель и скалер успешно загружены")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")

# 2. Описание входных данных (оставляем как было)
class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.post("/predict")
def predict_default(data: ClientData):
    # Превращаем данные в DataFrame, чтобы соблюсти порядок колонок
    df = pd.DataFrame([data.dict()])
    
    # 3. Масштабирование признаков (обязательно для нейросети!)
    # Нейросеть обучена на нормализованных данных, поэтому просто подать числа нельзя
    features_scaled = scaler.transform(df).astype(np.float32)
    
    # 4. Предсказание через ONNX
    # Результат ONNX всегда возвращается в виде списка массивов
    onnx_inputs = {input_name: features_scaled}
    onnx_output = session.run(None, onnx_inputs)
    
    # Извлекаем вероятность (выход слоя Sigmoid)
    probability = float(onnx_output[0][0][0])
    prediction = 1 if probability > 0.5 else 0
    
    return {
        "default_prediction": prediction,
        "default_probability": round(probability, 4),
        "engine": "ONNX Runtime"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ONNX"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)