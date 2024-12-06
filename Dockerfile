# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the project code
COPY . .

# Ensure scripts are executable
RUN chmod +x /app/run_pipeline.sh
RUN chmod +x /app/src/graph_processing/*.py

# Define the default command to run the pipeline
CMD ["./run_pipeline.sh"]