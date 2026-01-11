FROM python:3.10-slim

# Install system dependencies required by ML libs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade build tools
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy rest of the code
COPY . .

# Expose port
EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]
