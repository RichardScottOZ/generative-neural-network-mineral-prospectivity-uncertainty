# Dockerfile for Mineral Prospectivity AWS Batch Jobs

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command (will be overridden by batch job)
CMD ["python", "-m", "mineral_prospectivity.aws_batch.train_worker", "--help"]
