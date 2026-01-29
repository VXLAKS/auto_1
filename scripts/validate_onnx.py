import torch
import onnxruntime as ort
import numpy as np
import sys
import os

# Добавляем путь к src, чтобы импортировать архитектуру сети
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.train_nn import CreditNet

def validate_conversion():
    print("Начинаем валидацию ONNX модели...")
    
    # 1. Параметры (должны совпадать с теми, что были при обучении)
    input_size = 23 # количество признаков в датасете UCI Credit Card
    onnx_path = "models/model.onnx"
    pth_path = "models/model_nn.pth"

    if not os.path.exists(onnx_path) or not os.path.exists(pth_path):
        print("Ошибка: Файлы моделей не найдены. Сначала запусти src/models/train_nn.py")
        return

    # 2. Загружаем исходную PyTorch модель
    model = CreditNet(input_size)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    # 3. Загружаем ONNX сессию
    ort_session = ort.InferenceSession(onnx_path)

    # 4. Создаем случайные входные данные для теста
    dummy_input = np.random.randn(1, input_size).astype(np.float32)

    # 5. Получаем предсказание от PyTorch
    with torch.no_grad():
        torch_out = model(torch.from_numpy(dummy_input)).numpy()

    # 6. Получаем предсказание от ONNX
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_out = ort_outs[0]

    # 7. Сравниваем результаты
    # Используем малую погрешность (atol), так как типы данных могут чуть-чуть отличаться
    try:
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)
        print("Валидация пройдена! Предсказания PyTorch и ONNX идентичны.")
    except AssertionError as e:
        print(f"Валидация провалена: результаты различаются.\n{e}")

if __name__ == "__main__":
    validate_conversion()