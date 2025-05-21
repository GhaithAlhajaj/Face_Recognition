FROM python:3.9
RUN apt-get update && apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
COPY encodings.pkl .
COPY labels.pkl .
CMD ["streamlit", "run", "app.py", "--server.port=8080"]
