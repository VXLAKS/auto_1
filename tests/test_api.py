from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/") # Если есть ручка "/"
    assert response.status_code in [200, 404] 

# Добавь простой тест на загрузку модели или предобработку