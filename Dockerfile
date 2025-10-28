# Dockerfile
FROM python:3.11-slim

# Install system deps required by spaCy and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (leverages docker cache)
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download the small spaCy model at build time so the container is self-contained
RUN python -m spacy download en_core_web_sm

# Copy app code
COPY . /app

# Expose port
EXPOSE 8000

# Use a non-root user in production (optional)
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

# Start uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
