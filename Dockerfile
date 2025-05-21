FROM python:3.11-slim

# Set environment variables to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages for building dlib and running apps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-system-dev \
    libboost-thread-dev \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setup tools
RUN pip install --upgrade pip setuptools wheel

# Install dlib (source build to ensure compatibility with Python 3.11)
RUN pip install dlib==19.24.2

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
