# Install system packages required for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Try binary install first, fallback to source if it fails
RUN pip install dlib==19.24.2 --only-binary :all: || pip install dlib==19.24.2

# Install other Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
