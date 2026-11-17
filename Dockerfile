FROM python:3.12-slim

# System libraries OpenCV needs to decode video
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit needs a writable home on Spaces
ENV HOME=/tmp \
    OMP_NUM_THREADS=2 \
    DT_THREADS=2

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.maxUploadSize=200", \
     "--browser.gatherUsageStats=false"]
