
# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy everything needed for the app
COPY ./backend /app/backend
COPY ./recommender /app/recommender
COPY ./data /app/data
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Run FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
