FROM python:3.11-slim

WORKDIR /app

# Install pip deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Pre-download model weights during build
COPY app_preload.py /app/app_preload.py
RUN python app_preload.py && rm app_preload.py

# Remove build-time files
RUN rm -f requirements.txt Dockerfile

CMD ["python", "app.py"]
