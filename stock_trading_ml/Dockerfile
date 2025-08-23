# ML Trading Strategy Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/results /app/plots /app/saved_models /app/backtest_results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for any web interfaces (if needed in future)
EXPOSE 8000

# Default command - run the main script
CMD ["python", "main.py", "--config", "config.json", "--output", "results/"]

# Alternative commands:
# Run CLI: docker run -it ml-trading-strategy python cli.py --help
# Run tests: docker run ml-trading-strategy python -m pytest tests/ -v
# Quick test: docker run ml-trading-strategy python cli.py quick-test --ticker AAPL
# Interactive shell: docker run -it ml-trading-strategy /bin/bash
