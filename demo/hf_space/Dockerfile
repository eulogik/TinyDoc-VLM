FROM python:3.11-slim

WORKDIR /app

# Install pip deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Remove requirements.txt and Dockerfile from runtime
RUN rm requirements.txt Dockerfile

CMD ["python", "app.py"]
