FROM python:3.11-slim

WORKDIR /app

# Install pip deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Remove build-time files
RUN rm -f requirements.txt Dockerfile app_preload.py

CMD ["python", "app.py"]
