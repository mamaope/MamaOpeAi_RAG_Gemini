FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for faiss-cpu
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

EXPOSE 8090

# Run the uvicorn application server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
