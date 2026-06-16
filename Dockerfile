# Використовуємо легковаговий офіційний образ Python
FROM python:3.11-slim

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Окремо копіюємо requirements, щоб кешувати шари Docker
COPY requirements.txt .

# Встановлюємо залежності без збереження кешу pip (зменшуємо розмір образу)
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо наш скрипт
COPY fetch_data.py .

# За замовчуванням створюємо папку для даних всередині контейнера
RUN mkdir data

# Вказуємо дефолтну змінну оточення
ENV STOCK_TICKER="AAPL"

# Точка входу — запуск нашого скрипту
CMD ["python", "fetch_data.py"]
