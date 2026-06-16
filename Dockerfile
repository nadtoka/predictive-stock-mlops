FROM python:3.11-slim

WORKDIR /app

# Встановлюємо системні залежності (якщо знадобляться для компіляції деяких ліб)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо залежності та інсталюємо їх
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код проєкту
COPY . .

# Створюємо папки під дані та моделі всередині контейнера
RUN mkdir -p data models

# За замовчуванням запускаємо збір даних, але це можна перевизначити при старті контейнера
CMD ["python", "fetch_data.py"]
