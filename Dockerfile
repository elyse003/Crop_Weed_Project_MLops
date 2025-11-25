FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy Code
COPY . .

# Expose API Port
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]