import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize():
    # Пути к файлам
    model_fp32 = "models/model.onnx"
    model_quant = "models/model_quantized.onnx"
    
    print("🚀 Запуск процесса квантования...")

    # Проверка наличия исходной модели
    if not os.path.exists(model_fp32):
        print(f"❌ Ошибка: Файл {model_fp32} не найден. Сначала запусти src/models/train_nn.py")
        return

    try:
        # Выполняем динамическое квантование
        # Параметр extra_options с 'ForceQuantizeNoPreprocess' отключает 
        # встроенный shape inference, который вызывал ошибку (23) vs (64)
        quantize_dynamic(
            model_input=model_fp32,
            model_output=model_quant,
            weight_type=QuantType.QUInt8,
            extra_options={'ForceQuantizeNoPreprocess': True}
        )
        
        if os.path.exists(model_quant):
            print("✅ Квантование успешно завершено!")
            
            # Статистика по размерам для отчета
            size_old = os.path.getsize(model_fp32) / 1024
            size_new = os.path.getsize(model_quant) / 1024
            
            print("-" * 30)
            print(f"📊 Исходная модель (FP32): {size_old:.2f} KB")
            print(f"📊 Квантованная модель (INT8): {size_new:.2f} KB")
            print(f"📉 Сжатие: {size_old / size_new:.2f}x")
            print("-" * 30)
            print(f"Файл сохранен: {model_quant}")
        else:
            print("❌ Ошибка: Файл квантованной модели не был создан.")

    except Exception as e:
        print(f"❌ Произошла ошибка во время квантования: {e}")
        print("\nСовет: Если ошибка 'ShapeInferenceError' сохраняется, проверьте версию opset в train_nn.py (рекомендуется 14).")

if __name__ == "__main__":
    quantize()