FROM python:3.9-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY src/ src/
COPY models/ models/

# Запускаем API
CMD ["python", "src/api/app.py"]