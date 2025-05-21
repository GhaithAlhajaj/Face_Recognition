# Use a fuller Python image (not slim) for better compatibility and fewer memory issues
FROM python:3.10

# Install necessary system packages to build dlib and support GUI (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files to container
COPY . .

# Upgrade pip and install dependencies (try binary dlib first, fallback to source)
RUN pip install --upgrade pip setuptools wheel \
 && pip install dlib==19.24.2 --only-binary :all: || pip install dlib==19.24.2 \
 && pip install -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

