FROM python:3.12-slim

WORKDIR /app

# Install with the server extra so `vox serve` works out of the box.
COPY pyproject.toml README.md ./
COPY vox_agent ./vox_agent
COPY web ./web
RUN pip install --no-cache-dir ".[server]"

ENV VOX_STT=mock VOX_LLM=mock VOX_TTS=mock
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["vox", "serve", "--host", "0.0.0.0", "--port", "8000"]
