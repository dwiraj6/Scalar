FROM python:3.11-slim

# Metadata for HF Spaces
LABEL org.opencontainers.image.title="Email Triage OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv environment for email triage tasks"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py tasks.py graders.py environment.py server.py inference.py openenv.yaml ./

# HF Spaces runs on port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
