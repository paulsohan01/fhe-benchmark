# ----------  TenSEAL/Dockerfile  ----------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1️⃣ install apt deps for TenSEAL build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ cmake git wget libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# copy project
COPY . .

# small entrypoints decide role = client / server
COPY entrypoint_client.sh /entrypoint_client.sh
COPY entrypoint_server.sh /entrypoint_server.sh
RUN chmod +x /entrypoint_client.sh /entrypoint_server.sh
