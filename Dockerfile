# Stage 1: Сборка зависимостей
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Финальный легкий образ
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Копируем только нужное
COPY src/ ./src
COPY models/model.onnx ./models/model.onnx
COPY models/scaler.pkl ./models/scaler.pkl

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]