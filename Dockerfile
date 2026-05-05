FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

# Install system deps (split to avoid parallel build issues)
RUN apt-get update
RUN apt-get install -y --no-install-recommends curl build-essential
RUN rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

# Copy the app
COPY app ./app
COPY streamlit_app ./streamlit_app
COPY dataset ./dataset

ENV APP_ENV=production APP_NAME="Tree Evaluator API" APP_VERSION=0.1.0

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
