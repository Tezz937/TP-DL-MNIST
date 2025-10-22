FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir tensorflow==2.20.0 flask numpy



COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
