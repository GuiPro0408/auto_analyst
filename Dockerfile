FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    AUTO_ANALYST_LLM_BACKEND=gemini \
    AUTO_ANALYST_LOG_LEVEL=INFO \
    AUTO_ANALYST_LOG_REDACT_QUERIES=true \
    AUTO_ANALYST_LOG_FORMAT=plain \
    AUTO_ANALYST_FETCH_CONCURRENCY=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.space.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.space.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860

CMD ["chainlit", "run", "ui/chainlit_app.py", "--host", "0.0.0.0", "--port", "7860"]
