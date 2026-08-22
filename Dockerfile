# Use an official Python runtime as a parent image
# Using a slim version to keep the image size down
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files to disk
# and to ensure stdout/stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV, scientific libraries, and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Upgrade pip first to prevent hash mismatch errors, then install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5000 for the Flask server
EXPOSE 5000

# Set the HuggingFace cache directory to a known location inside the container
ENV HF_HOME=/app/.cache/huggingface

# Command to run the application
CMD ["python", "live/server.py"]
